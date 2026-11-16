import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# ---------------------------------------------------------------------------
# Escalas de tempo
# ---------------------------------------------------------------------------
# escala -> (regra de resample do pandas, períodos por ano)
# "semestral" usa os fechamentos de jun/dez (semestres de calendário).
ESCALAS = {
    "diaria": (None, 252),
    "mensal": ("ME", 12),
    "trimestral": ("QE", 4),
    "semestral": ("QE", 2),
    "anual": ("YE", 1),
}
ROTULOS = {
    "diaria": "diário",
    "mensal": "mensal",
    "trimestral": "trimestral",
    "semestral": "semestral",
    "anual": "anual",
}


def vis_preco_historico_total(df,normalizar = True):
    dataframe = _extrair_close(df)
    if normalizar:
        dataframe = (dataframe/dataframe.iloc[0]) * 100
        dataframe.plot(figsize = [15,8])
    else:
        dataframe.plot(figsize=[15,8])
    plt.xlabel('Data',  fontsize=21)
    plt.ylabel('Preço de Fechamento (normalizado)',fontsize=21)
    #plt.yscale('log')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc = 'best', fontsize=14);
    plt.show()
    return None


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def _extrair_close(df):
    if isinstance(df.columns, pd.MultiIndex) and "Close" in df.columns.levels[0]:
        df = df["Close"]
    return df


def _checar_escala(escala):
    if escala not in ESCALAS:
        raise ValueError(
            f"Escala '{escala}' inválida. Use uma de: {list(ESCALAS)}."
        )


def converter_taxa(taxa_anual, escala):
    """Converte uma taxa anual para a escala escolhida (juros compostos)."""
    _checar_escala(escala)
    per_ano = ESCALAS[escala][1]
    return (1 + taxa_anual) ** (1 / per_ano) - 1


def precos_na_escala(df, escala="anual"):
    """Preços de fechamento amostrados na escala de tempo escolhida."""
    _checar_escala(escala)
    df = _extrair_close(df)
    regra, _ = ESCALAS[escala]
    if regra is None:
        return df
    precos = df.resample(regra).last()
    if escala == "semestral":
        precos = precos[precos.index.month.isin([6, 12])]
    return precos


def estimar_parametros(df, escala="anual"):
    """Retorno médio e matriz de covariância, ambos na MESMA escala de tempo."""
    precos = precos_na_escala(df, escala)
    ret = precos.pct_change(fill_method=None)
    n_obs = int(ret.dropna(how="all").shape[0])
    if n_obs <= precos.shape[1]:
        warnings.warn(
            f"Escala '{escala}': só {n_obs} observações para {precos.shape[1]}"
            " ativos. A matriz de covariância fica singular/pouco confiável."
        )
    return (
        ret.mean().to_numpy(),
        ret.cov().to_numpy(),
        list(precos.columns),
        n_obs,
    )


# ---------------------------------------------------------------------------
# Geração de pesos e Monte Carlo
# ---------------------------------------------------------------------------
def gerar_pesos_validos(
    n_atv,
    num_pontos=500000,
    min_weight=0.01,
    max_weight=0.90,
    lote_inicial=1000000,
    max_iter=100,
    seed=None,
):
    """Gera pesos aleatórios (soma 1) respeitando os limites min/max.

    Gerar os pesos separadamente permite REUTILIZAR exatamente os mesmos
    portfólios candidatos em escalas/períodos diferentes, de modo que o ruído
    do Monte Carlo não se misture com o efeito que se quer medir.
    """
  
    if min_weight > max_weight:
        raise ValueError(
            f"O peso mínimo ({min_weight}) não pode ser maior que o peso máximo"
            f" ({max_weight})."
        )
    if min_weight * n_atv > 1.0:
        raise ValueError(
            f"Erro de viabilidade: O peso mínimo ({min_weight}) multiplicado por"
            f" {n_atv} ativos ultrapassa 100%."
        )
    if max_weight * n_atv < 1.0:
        raise ValueError(
            f"Erro de viabilidade: O peso máximo ({max_weight}) multiplicado por"
            f" {n_atv} ativos não é suficiente para atingir 100%."
        )

    rng = np.random.default_rng(seed)
    pesos_validos = []
    coletados = 0
    iteracao = 0

    while coletados < num_pontos:
        if iteracao >= max_iter:
            raise RuntimeError(
                f"Limite máximo de {max_iter} iterações atingido. Apenas"
                f" {coletados} de {num_pontos} portfólios válidos foram"
                " coletados. Suas restrições de min/max estão excessivamente"
                " restritivas para o número de ativos."
            )
        pesos_raw = rng.uniform(0, 1, size=(lote_inicial, n_atv))
        pesos = pesos_raw / np.sum(pesos_raw, axis=1, keepdims=True)
        mask = np.all((pesos >= min_weight) & (pesos <= max_weight), axis=1)
        validos = pesos[mask]
        if len(validos) > 0:
            pesos_validos.append(validos)
            coletados += len(validos)
        iteracao += 1

    return np.vstack(pesos_validos)[:num_pontos]


def monte_carlo_de_portifolios(
    df,
    escala="anual",
    pesos=None,
    seed=None,
    num_pontos=500000,
    **kw_pesos,
):
    """Monte Carlo de portfólios na escala de tempo escolhida.

    Retorno e Risco ficam na própria escala (ex.: escala="mensal" -> retorno
    médio mensal e volatilidade mensal, sem anualizar).
    Se `pesos` for passado, usa exatamente esses pesos (shape: n_pontos x n_ativos).
    `kw_pesos` (min_weight, max_weight, lote_inicial, max_iter) vai para
    gerar_pesos_validos.
    """
    df = _extrair_close(df)
    er_np, cov_np, ativos, _ = estimar_parametros(df, escala)

    if pesos is None:
        pesos = gerar_pesos_validos(
            len(ativos), num_pontos=num_pontos, seed=seed, **kw_pesos
        )
    elif pesos.shape[1] != len(ativos):
        raise ValueError("`pesos` tem número de colunas diferente do de ativos.")

    p_ret = pesos @ er_np
    var = np.sum((pesos @ cov_np) * pesos, axis=1)
    p_vol = np.sqrt(var)

    data = {"Retorno": p_ret, "Risco": p_vol}
    for i, symbol in enumerate(ativos):
        data["peso " + symbol] = pesos[:, i]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Seleção de portfólios de interesse
# ---------------------------------------------------------------------------
def selecionar_portfolios(
    portfolios,
    escala="anual",
    rf_poupanca=0.0829,
    rf_cdi=0.1423,
    rf_ipca=0.0463,
    retorno_alvo=0.14,
):
    """Portfólios de destaque. As taxas (rf_* e retorno_alvo) são ANUAIS e
    são convertidas internamente para a escala escolhida."""
    rf = {
        "poupanca": converter_taxa(rf_poupanca, escala),
        "cdi": converter_taxa(rf_cdi, escala),
        "ipca": converter_taxa(rf_ipca, escala),
    }
    resultados = {"min_vol": portfolios.loc[portfolios["Risco"].idxmin()]}
    for nome, taxa in rf.items():
        sharpe = (portfolios["Retorno"] - taxa) / portfolios["Risco"]
        resultados[nome] = portfolios.loc[sharpe.idxmax()]

    if retorno_alvo is not None:
        alvo = converter_taxa(retorno_alvo, escala)
        candidatos = portfolios[portfolios["Retorno"] >= alvo]
        if candidatos.empty:
            print(
                "Aviso: O retorno desejado é maior que o retorno máximo"
                " simulado. Retornando o portfólio de maior retorno."
            )
            resultados["portfolio_alvo"] = portfolios.loc[
                portfolios["Retorno"].idxmax()
            ]
        else:
            resultados["portfolio_alvo"] = portfolios.loc[
                candidatos["Risco"].idxmin()
            ]
    return resultados


def extrair_fronteira(portfolios, n_pontos=15):
    """Pontos da fronteira eficiente: o portfólio de menor risco em cada uma
    de `n_pontos` faixas de retorno, do mínimo risco até o maior retorno."""
    r_min = portfolios.loc[portfolios["Risco"].idxmin(), "Retorno"]
    r_max = portfolios["Retorno"].max()
    cand = portfolios[portfolios["Retorno"] >= r_min]
    faixas = pd.cut(
        cand["Retorno"],
        bins=np.linspace(r_min, r_max, n_pontos + 1),
        include_lowest=True,
    )
    idx = cand.groupby(faixas, observed=True)["Risco"].idxmin()
    return portfolios.loc[idx.values].sort_values("Retorno")


def _tabela_portfolios(resultados, fronteira=None, **meta):
    linhas = [{"tipo": nome, **p.to_dict()} for nome, p in resultados.items()]
    if fronteira is not None:
        for i, (_, p) in enumerate(fronteira.iterrows()):
            linhas.append({"tipo": f"fronteira_{i:02d}", **p.to_dict()})
    tabela = pd.DataFrame(linhas)
    for chave, valor in reversed(list(meta.items())):
        tabela.insert(0, chave, valor)
    return tabela


def resumo_pesos(tabela, tipo="min_vol", por="escala"):
    """Pesos de um tipo de portfólio, uma linha por escala (ou por período)."""
    cols = [c for c in tabela.columns if c.startswith("peso ")]
    return tabela[tabela["tipo"] == tipo].set_index(por)[cols]


# ---------------------------------------------------------------------------
# (2) Simular e salvar
# ---------------------------------------------------------------------------
def simular_e_salvar(
    df,
    escala="anual",
    caminho="dados/portfolios_eficientes.csv",
    n_fronteira=15,
    rf_poupanca=0.0829,
    rf_cdi=0.1423,
    rf_ipca=0.0463,
    retorno_alvo=0.20,
    seed=None,
    **kw_pesos,
):
    """Roda a simulação e salva em CSV os portfólios de interesse
    (min_vol, poupanca, cdi, ipca, portfolio_alvo) + `n_fronteira` pontos da
    fronteira. Retorna a tabela salva."""
    portfolios = monte_carlo_de_portifolios(
        df, escala=escala, seed=seed, **kw_pesos
    )
    resultados = selecionar_portfolios(
        portfolios, escala, rf_poupanca, rf_cdi, rf_ipca, retorno_alvo
    )
    fronteira = extrair_fronteira(portfolios, n_fronteira) if n_fronteira else None
    tabela = _tabela_portfolios(resultados, fronteira, escala=escala)
    if caminho is not None:
        tabela.to_csv(caminho, index=False)
    return tabela


# ---------------------------------------------------------------------------
# (1) Comparar escalas de tempo
# ---------------------------------------------------------------------------
def comparar_escalas(
    df,
    escalas=("diaria", "mensal", "trimestral", "semestral", "anual"),
    caminho=None,
    n_fronteira=0,
    rf_poupanca=0.0829,
    rf_cdi=0.1423,
    rf_ipca=0.0463,
    retorno_alvo=0.14,
    seed=0,
    **kw_pesos,
):
    """Mesmos ativos, mesmo período e MESMOS pesos candidatos, variando só a
    escala de tempo. Retorna tabela longa (coluna `escala`); use
    resumo_pesos(tabela, "min_vol") para ver os pesos lado a lado."""
    df = _extrair_close(df)
    pesos = gerar_pesos_validos(df.shape[1], seed=seed, **kw_pesos)

    tabelas = []
    for escala in escalas:
        portfolios = monte_carlo_de_portifolios(df, escala=escala, pesos=pesos)
        resultados = selecionar_portfolios(
            portfolios, escala, rf_poupanca, rf_cdi, rf_ipca, retorno_alvo
        )
        fronteira = (
            extrair_fronteira(portfolios, n_fronteira) if n_fronteira else None
        )
        tabelas.append(_tabela_portfolios(resultados, fronteira, escala=escala))

    tabela = pd.concat(tabelas, ignore_index=True)
    if caminho is not None:
        tabela.to_csv(caminho, index=False)
    return tabela


# ---------------------------------------------------------------------------
# (3) Comparar intervalos de tempo
# ---------------------------------------------------------------------------
def comparar_periodos(
    df,
    periodos,
    escala="anual",
    caminho=None,
    n_fronteira=0,
    rf_poupanca=0.0829,
    rf_cdi=0.1423,
    rf_ipca=0.0463,
    retorno_alvo=0.14,
    seed=0,
    **kw_pesos,
):
    """Mesmos ativos, mesma escala e mesmos pesos candidatos, variando só o
    intervalo de datas.

    periodos: dict {"nome": ("inicio", "fim")}, ex.:
        {"2015-2019": ("2015-01-01", "2019-12-31"),
         "2020-2024": ("2020-01-01", "2024-12-31")}
    """
    df = _extrair_close(df)
    pesos = gerar_pesos_validos(df.shape[1], seed=seed, **kw_pesos)

    tabelas = []
    for nome, (inicio, fim) in periodos.items():
        recorte = df.loc[inicio:fim]
        if recorte.empty:
            raise ValueError(f"Período '{nome}' não tem dados.")
        portfolios = monte_carlo_de_portifolios(
            recorte, escala=escala, pesos=pesos
        )
        resultados = selecionar_portfolios(
            portfolios, escala, rf_poupanca, rf_cdi, rf_ipca, retorno_alvo
        )
        fronteira = (
            extrair_fronteira(portfolios, n_fronteira) if n_fronteira else None
        )
        tabelas.append(
            _tabela_portfolios(
                resultados, fronteira, periodo=nome, escala=escala,
                inicio=recorte.index[0].date(), fim=recorte.index[-1].date(),
            )
        )

    tabela = pd.concat(tabelas, ignore_index=True)
    if caminho is not None:
        tabela.to_csv(caminho, index=False)
    return tabela


# ---------------------------------------------------------------------------
# Gráfico
# ---------------------------------------------------------------------------
def plotar_fronteira_eficiente(
    df,
    escala="anual",
    portfolios=None,
    rf_poupanca=0.0829,
    rf_cdi=0.1423,
    rf_ipca=0.0463,
    grau=2,
    retorno_alvo=0.14,
    seed=None,
    **kw_pesos,
):
    """Calcula os portfólios de destaque, realiza o ajuste polinomial,
    destaca opcionalmente um portfólio alvo por retorno e plota o gráfico.

    rf_* e retorno_alvo são taxas ANUAIS (convertidas para a escala).
    Se `portfolios` (saída de monte_carlo_de_portifolios) for passado, não
    roda a simulação de novo.
    """
    if portfolios is None:
        portfolios = monte_carlo_de_portifolios(
            df, escala=escala, seed=seed, **kw_pesos
        )

    resultados = selecionar_portfolios(
        portfolios, escala, rf_poupanca, rf_cdi, rf_ipca, retorno_alvo
    )
    min_vol_port = resultados["min_vol"]
    opt_ris_por0 = resultados["poupanca"]
    opt_ris_por1 = resultados["cdi"]
    opt_ris_por2 = resultados["ipca"]
    portfolio_alvo = resultados.get("portfolio_alvo")

    # Ajuste polinomial (fit) da Fronteira Eficiente
    pontos = list(resultados.values())
    x = np.array([p["Risco"] for p in pontos])
    y = np.array([p["Retorno"] for p in pontos])
    z = np.polyfit(y, x, grau)  # x em função de y
    f = np.poly1d(z)
    amp = y.max() - y.min()  # margem proporcional (independe da escala)
    y_new = np.linspace(y.min() - 0.5 * amp, y.max() + 0.2 * amp, 200)
    x_new = f(y_new)

    rot = ROTULOS[escala]
    plt.figure(figsize=(15, 8))
    s1 = 400
    plt.scatter(
        portfolios["Risco"], portfolios["Retorno"],
        c="g", marker="o", s=50, alpha=0.4, label="Portfólios Simulados",
    )
    plt.scatter(
        min_vol_port["Risco"], min_vol_port["Retorno"],
        c="b", marker="v", s=s1, alpha=0.75, label="Risco Mínimo",
    )
    plt.scatter(
        opt_ris_por0["Risco"], opt_ris_por0["Retorno"],
        c="m", marker="o", s=s1, alpha=0.75, label="Fator de Risco (Poupança)",
    )
    plt.scatter(
        opt_ris_por1["Risco"], opt_ris_por1["Retorno"],
        c="k", marker="s", s=s1, alpha=0.75, label="Fator de Risco (CDI)",
    )
    plt.scatter(
        opt_ris_por2["Risco"], opt_ris_por2["Retorno"],
        c="orange", marker="*", s=s1, alpha=1, label="Fator de Risco (IPCA)",
    )
    if portfolio_alvo is not None:
        plt.scatter(
            portfolio_alvo["Risco"], portfolio_alvo["Retorno"],
            c="cyan", marker="P", s=s1 + 100, edgecolors="black",
            linewidth=1.5, alpha=1,
            label=f"Alvo (Retorno anual ≥ {retorno_alvo*100:.1f}%)",
        )
    plt.plot(
        x_new, y_new, "--", c="k", linewidth=3, alpha=0.75,
        label="Fronteira Eficiente",
    )
    plt.xlabel(f"Risco (Volatilidade {rot})", fontsize=22)
    plt.ylabel(f"Retorno Esperado {rot}", fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=16)
    plt.show()
    return resultados