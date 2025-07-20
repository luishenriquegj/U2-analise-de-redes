import wikipedia
import networkx as nx
import streamlit as st

def get_wikipedia_links(page_title):
    try:
        wikipedia.set_lang("pt")
        page = wikipedia.page(page_title, auto_suggest=False)
        return page.links
    except wikipedia.exceptions.PageError:
        st.warning(f"Página '{page_title}' não encontrada na Wikipedia. Verifique a grafia.")
        return []
    except wikipedia.exceptions.DisambiguationError as e:
        st.info(f"Página '{page_title}' é uma desambiguação. Tentando resolver...")
        if not e.options or e.options[0].strip().lower() == page_title.strip().lower():
            st.warning(f"Desambiguação para '{page_title}' complexa ou sem opção clara. Ignorando este termo.")
            return []
        try:
            resolved_page = wikipedia.page(e.options[0], auto_suggest=False)
            st.info(f"Desambiguação resolvida para '{resolved_page.title}'.")
            return resolved_page.links
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            st.warning(f"Falha ao resolver a primeira opção de desambiguação para '{page_title}' ('{e.options[0]}'). Ignorando este termo.")
            return []
        except Exception as inner_e:
            st.error(f"Erro inesperado ao tentar resolver desambiguação para '{page_title}': {inner_e}")
            return []
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao buscar a página: {e}")
        return []

def build_network_from_wikipedia(seed_page_title, max_nodes=100, max_depth=2):
    G = nx.Graph()
    pages_to_visit = [(seed_page_title, 0)] 
    visited_pages = set()

    # ---  LISTA DE EXCLUSÃO  ---
    STOP_WORDS = [
        'Anexo:', 'Portal:', 'Ajuda:', 'Usuário:', 'Ficheiro:',
        'Categoria:', 'Discussão:', 'Especial:', 'Wikipedia:','fevereiro','outubro',
        'março','janeiro','julho','maio','junho','abril','agosto','dezembro','novembro','setembro', '20th Century Fox', 'Alcunha','Atari',
        'França','AdoroCinema', 'Estados Unidos','Prêmio','IMDb', 'Rotten Tomatoes','Cinema','Japão','DVD','Ficção Científica', 'Roterista','Brasil',
        'Space Opera', 'Roterista','Internacional','Ator','Wikimedia','WikiTree','MusicBrainz','Akira Kurosawa','Alec Guinness','Suécia',
        'Action Figure','Marketing','Alemanha','MTV','Marvel','Filme','3D','minuto','Adam Driver','American','Alien','Iwo Jima','IGN','Alter-ego','TV',
        'Arcade','Banda desenhada','A fortaleza escondida','Amor','Série','Disney','Antologia', 'Box office','Animação','Blockbuster',
        'Billie Lourd','Ahmed Best','Dolar','Dólar','Produtor','CNN','.com','mark hamill', 'harrison ford', 'carrie fisher', 'alec guinness', 
        'peter mayhew', 'Agência Nacional','Blu-Ray','América do Sul','david prowse', 'james earl jones', 'anthony daniels', 'kenny baker',
        'peter cushing', 'Biblioteca Nacional','billy dee williams', 'ian mcdiarmid', 'frank oz', 'sebastian shaw', 'denis lawson','warwick davis',
        'liam neeson', 'ewan mcgregor', 'natalie portman', 'jake lloyd', 'hayden christensen',
        'samuel l. jackson', 'christopher lee', 'pernilla august', 'temuera morrison','jimmy smits',
        'daisy ridley', 'john boyega', 'adam driver', 'oscar isaac', 'andy serkis', 'BBC',
        'lupita', 'domhnall gleeson', 'gwendoline christie', 'max von sydow',
        'kelly marie tran', 'laura dern', 'benicio del toro',
        'felicity jones', 'diego luna', 'ben mendelsohn', 'donnie yen', 'mads mikkelsen',
        'alan tudyk', 'riz ahmed', 'forest whitaker', 'alden ehrenreich', 'woody harrelson',
        'emilia clarke', 'donald glover', 'phoebe waller-bridge', 'paul bettany'
    ]
    
    while pages_to_visit and len(G.nodes) < max_nodes:
        current_page_title, current_depth = pages_to_visit.pop(0)

        if current_page_title in visited_pages:
            continue
        
        # --- APLICA O FILTRO DE PROFUNDIDADE ---
        if current_depth >= max_depth:
            continue

        # Adiciona o nó atual ao grafo e à lista de visitados
        normalized_current_page = current_page_title.lower().strip()
        G.add_node(normalized_current_page, label=current_page_title)
        visited_pages.add(current_page_title)

        links = get_wikipedia_links(current_page_title)
        
        for link in links:
            # Normaliza o link para verificação
            normalized_link = link.lower().strip()

            # --- APLICA OS FILTROS  ---
            # 1. Ignora se for um número (geralmente um ano)
            if normalized_link.isnumeric():
                continue
            # 2. Ignora se corresponder a um padrão da lista de exclusão
            if any(pattern.lower().strip() in normalized_link for pattern in STOP_WORDS):
                continue
            # --- FIM DOS FILTROS ---

            # Se o link passou pelos filtros, adiciona ao grafo
            if len(G.nodes) >= max_nodes:
                break

            G.add_node(normalized_link, label=link)
            G.add_edge(normalized_current_page, normalized_link)

            if link not in visited_pages and not any(link == p[0] for p in pages_to_visit):
                pages_to_visit.append((link, current_depth + 1))
    
    return G