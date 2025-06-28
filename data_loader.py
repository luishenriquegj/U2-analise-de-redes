import wikipedia
import networkx as nx
import streamlit as st

def get_wikipedia_links(page_title):
    """
    Busca os links de uma página da Wikipedia.
    Define o idioma para português.
    Inclui tratamento aprimorado para DisambiguationError.
    """
    try:
        wikipedia.set_lang("pt")
        page = wikipedia.page(page_title, auto_suggest=False)
        return page.links
    except wikipedia.exceptions.PageError:
        st.warning(f"Página '{page_title}' não encontrada na Wikipedia. Verifique a grafia.")
        return []
    except wikipedia.exceptions.DisambiguationError as e:
        # Tenta a primeira opção da desambiguação.
        # Se a primeira opção for igual ao título original ou for vazia,
        # ou se a tentativa de carregar a primeira opção ainda der erro,
        # significa que a desambiguação é complexa ou o termo não tem uma página clara.
        st.info(f"Página '{page_title}' é uma desambiguação. Tentando resolver...")
        
        # Evita loops infinitos ou tentativas em termos vazios
        if not e.options or e.options[0].strip().lower() == page_title.strip().lower():
            st.warning(f"Desambiguação para '{page_title}' complexa ou sem opção clara. Ignorando este termo.")
            return []
        
        try:
            # Tenta carregar a primeira opção sugerida
            resolved_page = wikipedia.page(e.options[0], auto_suggest=False)
            st.info(f"Desambiguação resolvida para '{resolved_page.title}'.")
            return resolved_page.links
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            # Se a primeira opção ainda gerar PageError ou outra DisambiguationError,
            # consideramos que não há uma resolução clara e pulamos este link.
            st.warning(f"Falha ao resolver a primeira opção de desambiguação para '{page_title}' ('{e.options[0]}'). Ignorando este termo.")
            return []
        except Exception as inner_e:
            st.error(f"Erro inesperado ao tentar resolver desambiguação para '{page_title}': {inner_e}")
            return []
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao buscar a página: {e}")
        return []

import networkx as nx

def build_network_from_wikipedia(seed_page_title, max_nodes=100):
    G = nx.Graph()
    visited_pages = set()
    pages_to_visit = [(seed_page_title, 0)] # (page_title, current_depth)

    # Normalizar o seed_page_title e adicioná-lo
    normalized_seed_page_title = seed_page_title.lower().strip()
    G.add_node(normalized_seed_page_title, label=normalized_seed_page_title)
    visited_pages.add(normalized_seed_page_title)

    while pages_to_visit and len(G.nodes) < max_nodes:
        current_page_title, current_depth = pages_to_visit.pop(0)

        # Se a página já foi visitada (e já tem nó no grafo), continue para a próxima iteração.
        if current_page_title in visited_pages and current_page_title != normalized_seed_page_title:
             continue 

        visited_pages.add(current_page_title)

        links = get_wikipedia_links(current_page_title)
        for link in links:
            normalized_link = link.lower().strip()

            if len(G.nodes) >= max_nodes:
                break # Sai do loop interno 'for link in links' se o limite for atingido

            # Se o nó normalizado ainda não está no grafo, adiciona-o
            if normalized_link not in G.nodes: 
                G.add_node(normalized_link, label=normalized_link)

            G.add_edge(current_page_title, normalized_link)

            if normalized_link not in visited_pages and \
               not any(normalized_link == p[0] for p in pages_to_visit):
                pages_to_visit.append((normalized_link, current_depth + 1))

    return G