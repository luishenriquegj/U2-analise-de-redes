# app.py
import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components 
import pandas as pd
import matplotlib.pyplot as plt 

# Importa as funções do nosso módulo data_loader
from data_loader import build_network_from_wikipedia

# --- Funções de Visualização e Análise (visualize_network permanece a mesma) ---

def visualize_network(graph, seed_page_title="", graph_type="full"):
    """
    Gera o HTML da visualização interativa da rede usando Pyvis.
    Permite visualizar o grafo completo, o maior componente conectado, ou nós de alto grau.
    Retorna a string HTML para ser exibida pelo Streamlit.
    """
    if not graph.nodes:
        st.warning("Não há nós para visualizar no grafo.")
        return "" 

    display_graph = nx.Graph()

    if graph_type == "full":
        display_graph = graph
    elif graph_type == "largest_connected_component":
        if nx.is_empty(graph):
            st.warning("O grafo está vazio. Não é possível encontrar componentes conectados.")
            return "" 

        connected_comps_list = list(nx.connected_components(graph))

        if not connected_comps_list:
            st.warning("Nenhum componente conectado encontrado no grafo.")
            return "" 

        largest_cc = max(connected_comps_list, key=len)

        display_graph = graph.subgraph(largest_cc).copy()
        st.info(f"Visualizando o **Maior Componente Conectado** ({len(largest_cc)} nós).")
    elif graph_type == "high_degree_nodes":
        if nx.is_empty(graph):
            st.warning("O grafo está vazio. Não é possível encontrar nós de alto grau.")
            return "" 

        degree_dict = dict(graph.degree())
        num_high_degree_nodes = min(max(5, int(len(graph.nodes) * 0.10)), 50)
        
        top_nodes = sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)[:num_high_degree_nodes]
        high_degree_node_names = [node for node, degree in top_nodes]
        
        nodes_for_subgraph = set(high_degree_node_names)
        for node in high_degree_node_names:
            nodes_for_subgraph.update(graph.neighbors(node))
            
        display_graph = graph.subgraph(nodes_for_subgraph).copy()

        st.info(f"Visualizando um subgrafo focado nos **{len(high_degree_node_names)} nós de maior grau** e seus vizinhos diretos ({len(display_graph.nodes)} nós no total).")
    
    if not display_graph.nodes:
        st.warning("O subgrafo selecionado está vazio. Tente ajustar os parâmetros de rede ou selecione 'Grafo Completo'.")
        return "" 

    net = Network(notebook=True, height="750px", width="100%", directed=False, cdn_resources='remote')

    for node, attrs in display_graph.nodes(data=True):
        size = 20 if node == seed_page_title else 10
        title_text = f"Nó: {node}\nGrau: {display_graph.degree(node)}"
        net.add_node(node, label=attrs.get('label', node), title=title_text, size=size)

    for u, v in display_graph.edges():
        net.add_edge(u, v)

    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 12
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": {
          "forceDirection": "none"
        }
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "barnesHut",
        "timestep": 0.35
      }
    }
    """)

    try:
        html_content = net.generate_html()
        return html_content
    except Exception as e:
        st.error(f"Erro ao gerar o HTML da visualização da rede: {e}")
        return "" 

# A função analyze_network é atualizada para incluir as novas métricas de centralidade
def analyze_network(graph):
    """
    Realiza e exibe a análise estatística e estrutural da rede completa,
    incluindo distribuição de grau e diferentes métricas de centralidade.
    """
    if not graph.nodes:
        st.warning("Não há dados para análise. O grafo está vazio.")
        return

    st.subheader("Estatísticas Descritivas da Rede")
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    avg_degree = sum(dict(graph.degree()).values()) / num_nodes if num_nodes > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nós", num_nodes)
    with col2:
        st.metric("Arestas", num_edges)
    with col3:
        st.metric("Grau Médio", f"{avg_degree:.2f}")

    st.subheader("Métricas Estruturais da Rede")

    # 1. Densidade da Rede
    density = nx.density(graph)
    st.markdown(f"**Densidade da Rede:** `{density:.4f}`")
    st.markdown("""
    A **densidade** indica quão conectados os nós estão entre si, em comparação com o número máximo possível de conexões.
    Um valor próximo de 0 indica uma rede **esparsa** (poucas conexões); um valor próximo de 1 indica uma rede **densa** (muitas conexões).
    """)

    # 2. Assortatividade da Rede (Assortativity Coefficient)
    if num_nodes > 1 and num_edges > 0:
        assortativity = nx.degree_assortativity_coefficient(graph)
        st.markdown(f"**Coeficiente de Assortatividade de Grau:** `{assortativity:.4f}`")
        st.markdown("""
        A **assortatividade** mede a tendência de nós com graus semelhantes se conectarem.
        - Um valor **positivo** indica que nós de alto grau tendem a se conectar com outros nós de alto grau (redes 'ricas ficam mais ricas').
        - Um valor **negativo** indica que nós de alto grau tendem a se conectar com nós de baixo grau (redes 'centralizadas' ou 'estrela').
        - Um valor **próximo de zero** indica uma mistura aleatória.
        """)
    else:
        st.info("Não é possível calcular a Assortatividade para este grafo (poucos nós/arestas).")

    # 3. Coeficiente de Clustering Global (Global Clustering Coefficient)
    if num_nodes > 1 and num_edges > 0:
        avg_clustering = nx.average_clustering(graph)
        st.markdown(f"**Coeficiente de Clustering Global:** `{avg_clustering:.4f}`")
        st.markdown("""
        O **coeficiente de clustering global** indica a probabilidade de que os vizinhos de um nó também sejam vizinhos entre si,
        formando "triângulos" ou "cliques". Um valor alto sugere que a rede é composta por comunidades ou grupos coesos.
        """)
    else:
        st.info("Não é possível calcular o Coeficiente de Clustering para este grafo (poucos nós/arestas).")

    st.markdown("---")
    st.subheader("Componentes Conectados")
    
    # Componentes Fracamente Conectados (Weakly Connected Components - WCC)
    num_wcc = nx.number_connected_components(graph) 
    st.markdown(f"**Número de Componentes Conectados (WCCs):** `{num_wcc}`")
    st.markdown("""
    Um **Componente Conectado (WCC)** é um subgrafo onde cada nó pode ser alcançado a partir de qualquer outro nó dentro do mesmo subgrafo,
    ignorando a direção das arestas. Para grafos não dirigidos, este é o conceito fundamental de conectividade.
    """)
    if num_wcc > 0:
        largest_cc_size_for_wcc = max(len(c) for c in nx.connected_components(graph))
        st.markdown(f"**Tamanho do Maior Componente Conectado (WCC):** `{largest_cc_size_for_wcc}` nós")
    
    st.info("⚠️ **Componentes Fortemente Conectados (SCCs)** são aplicáveis a **grafos dirigidos**. A rede da Wikipedia que estamos gerando está sendo tratada como **não dirigida** neste contexto.")

    # --- Distribuição de Grau ---
    st.markdown("---")
    st.subheader("Distribuição de Grau da Rede")

    if num_nodes > 0:
        all_degrees = [degree for node, degree in graph.degree()]

        if all_degrees:
            min_degree = min(all_degrees)
            max_degree = max(all_degrees)

            # Ajuste dinâmico dos bins e ticks
            if min_degree == max_degree: # Caso todos os nós tenham o mesmo grau
                bins_range = [min_degree - 0.5, max_degree + 0.5]
                tick_values = [min_degree]
            elif max_degree - min_degree < 10: # Para poucos graus diferentes, mostrar todos
                bins_range = range(min_degree, max_degree + 2)
                tick_values = range(min_degree, max_degree + 1)
            else: # Para uma ampla gama de graus, usar um intervalo para os ticks
                bins_range = range(min_degree, max_degree + 2)
                # Tenta exibir cerca de 10 ticks, arredondando o intervalo
                ideal_num_ticks = 10
                step = max(1, (max_degree - min_degree) // ideal_num_ticks)
                tick_values = range(min_degree, max_degree + 1, step)


            fig, ax = plt.subplots()
            ax.hist(all_degrees, bins=bins_range, edgecolor='black', align='left')
            ax.set_title('Histograma da Distribuição de Grau')
            ax.set_xlabel('Grau do Nó')
            ax.set_ylabel('Número de Nós')
            ax.set_xticks(list(tick_values))

            st.pyplot(fig)
            plt.close(fig)

            st.markdown("""
            O **Histograma da Distribuição de Grau** mostra a frequência com que cada grau (número de conexões)
            aparece na rede.
            - Em redes como a Wikipedia (muitas vezes consideradas redes de mundo pequeno ou livres de escala),
              é comum ver muitos nós com poucos graus e poucos nós (os "hubs") com graus muito altos,
              resultando em uma distribuição com uma "cauda longa" para a direita.
            """)
        else:
            st.info("Não há graus para plotar (grafo sem nós ou arestas).")
    else:
        st.info("Não há nós na rede para calcular a distribuição de grau.")

    # --- Centralidade dos Nós ---
    st.markdown("---")
    st.subheader("Centralidade dos Nós")

    top_k = st.slider("Exibir até Top-200 Nós por Centralidade", 1, 200, 5)

    if num_nodes > 0 and num_edges > 0:
        # Centralidade de Grau (Degree Centrality)
        st.markdown("#### Centralidade de Grau")
        degree_centrality = nx.degree_centrality(graph)
        sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
        df_degree_topk = pd.DataFrame(sorted_degree, columns=['Nó', 'Centralidade de Grau'])
        st.dataframe(df_degree_topk)

        # Centralidade de Intermediação (Betweenness Centrality)
        st.markdown("#### Centralidade de Intermediação")
        betweenness_centrality = nx.betweenness_centrality(graph)
        sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
        df_betweenness_topk = pd.DataFrame(sorted_betweenness, columns=['Nó', 'Centralidade de Intermediação'])
        st.dataframe(df_betweenness_topk)

        # Centralidade de Proximidade (Closeness Centrality)
        st.markdown("#### Centralidade de Proximidade")
        if num_wcc > 1:
             st.info("Para grafos desconectados, a Centralidade de Proximidade é calculada apenas dentro do maior componente conectado.")
             largest_cc_nodes = max(nx.connected_components(graph), key=len)
             subgraph_for_closeness = graph.subgraph(largest_cc_nodes).copy()
             closeness_centrality = nx.closeness_centrality(subgraph_for_closeness)
        else:
            closeness_centrality = nx.closeness_centrality(graph)

        sorted_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
        df_closeness_topk = pd.DataFrame(sorted_closeness, columns=['Nó', 'Centralidade de Proximidade'])
        st.dataframe(df_closeness_topk)

        # Centralidade de Vetor Próprio (Eigenvector Centrality)
        st.markdown("#### Centralidade de Vetor Próprio")
        try:
            eigenvector_centrality = nx.eigenvector_centrality(graph)
            sorted_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
            df_eigenvector_topk = pd.DataFrame(sorted_eigenvector, columns=['Nó', 'Centralidade de Vetor Próprio'])
            st.dataframe(df_eigenvector_topk)

        except nx.NetworkXException as e:
            st.warning(f"Não foi possível calcular a Centralidade de Vetor Próprio: {e}. Isso pode ocorrer em grafos desconectados ou muito esparsos.")
    else:
        st.info("Não é possível calcular métricas de centralidade para um grafo vazio ou sem arestas.")


# --- Configurações do Streamlit e Interface do Usuário ---

st.set_page_config(page_title="Analisador de Redes da Wikipedia", layout="wide")

st.title("🌐 Analisador e Visualizador de Redes da Wikipedia")
st.markdown("""
Esta aplicação permite explorar redes complexas baseadas em páginas da Wikipedia.
Insira o título de uma página para visualizar suas conexões e obter análises estatísticas.
""")

if 'graph' not in st.session_state:
    st.session_state.graph = nx.Graph()
if 'seed_page_title' not in st.session_state:
    st.session_state.seed_page_title = "Python (linguagem de programação)"

with st.sidebar:
    st.header("Configurações da Rede")
    page_title_input = st.text_input("Título da Página da Wikipedia:", st.session_state.seed_page_title)
    

    max_nodes = st.slider("Número Máximo de Nós", 50, 2000, 500)

    st.markdown("---")
    st.header("Visualização da Rede")
    graph_display_option = st.radio(
        "Selecionar Subconjunto do Grafo:",
        ("Grafo Completo", "Maior Componente Conectado", "Nós de Maior Grau"),
        key="graph_display_option"
    )

    if st.button("Gerar Rede e Análise"):
        if page_title_input:
            st.session_state.seed_page_title = page_title_input
            with st.spinner("Construindo a rede... Isso pode levar um tempo para páginas grandes."):
                st.session_state.graph = build_network_from_wikipedia(
                    st.session_state.seed_page_title, max_nodes
                )
            if st.session_state.graph.number_of_nodes() > 0:
                st.success("Rede construída com sucesso!")
            else:
                st.warning("Não foi possível construir a rede. Verifique o título da página ou os parâmetros.")
        else:
            st.warning("Por favor, insira o título de uma página da Wikipedia.")
    
    st.markdown("---")
    st.info("""
    **Dicas:**
    - Aumentar o **Número Máximo de Nós** fará com que a busca por links explore mais camadas,
      mas também aumentará o tempo de processamento e o uso de memória.
    - Páginas muito populares podem gerar grafos enormes e lentos para renderizar.
    """)

if st.session_state.graph.number_of_nodes() > 0:
    st.write("---")
    st.subheader("Visualização da Rede")

    html_to_display = "" 
    if graph_display_option == "Grafo Completo":
        html_to_display = visualize_network(st.session_state.graph, st.session_state.seed_page_title, "full")
    elif graph_display_option == "Maior Componente Conectado":
        html_to_display = visualize_network(st.session_state.graph, st.session_state.seed_page_title, "largest_connected_component")
    elif graph_display_option == "Nós de Maior Grau":
        html_to_display = visualize_network(st.session_state.graph, st.session_state.seed_page_title, "high_degree_nodes")
    
    if html_to_display: 
        components.html(html_to_display, height=750)

    st.write("---")
    analyze_network(st.session_state.graph)