Clone o Repositório:
Bash

git clone <https://github.com/luishenriquegj/U2-analise-de-redes.git> 

Crie e Ative um Ambiente Virtual (recomendado para gerenciar dependências):
Bash

# Criar o ambiente virtual
python -m venv .venv

# Ativar o ambiente virtual:
# No Windows (PowerShell):
.venv\Scripts\Activate.ps1

# No Windows (Prompt de Comando - CMD):
.venv\Scripts\activate.bat

# No Linux / macOS / Git Bash:
source .venv/bin/activate

Instale as Dependências:
Com o ambiente virtual ativado, instale todas as bibliotecas necessárias:
Bash

    pip install streamlit wikipedia networkx pyvis matplotlib pandas

Execução da Aplicação

Com o ambiente virtual ativado, execute a aplicação Streamlit a partir do diretório raiz do projeto:
Bash

streamlit run app.py

Isso abrirá a aplicação em seu navegador web padrão (geralmente em http://localhost:8501).

Estrutura do Projeto

    app.py: Contém a interface principal da aplicação Streamlit, lógica de visualização e apresentação das análises.

    data_loader.py: Contém as funções responsáveis pela interação com a API da Wikipedia e pela construção do grafo NetworkX.

    network.html: Arquivo HTML gerado temporariamente pela Pyvis para exibir o grafo interativo.

Observações

    Desempenho: Redes com muitos nós (acima de 500-1000, dependendo da máquina) podem levar um tempo considerável para serem construídas e renderizadas, além de consumir bastante memória.

    Páginas de Desambiguação: O módulo wikipedia pode encontrar páginas de desambiguação. O script tenta resolver isso automaticamente, mas em casos complexos, ele pode pular links ambíguos para evitar travamentos.

    Grafos Não Dirigidos: A rede é construída como um grafo não dirigido (networkx.Graph). Métricas como Componentes Fortemente Conectados (SCCs) são mais relevantes para grafos dirigidos, e isso é observado na análise.

# Para rodar o a visualização do Gephi, voce precisará:

1. acessar a pasta Gephi dentro deste projeto com o CMD/terminal
2. rodar o comando python3 -m http.server   
        No seu terminal voce deve ver algo parecido com isso:
        \U2-analise-de-redes\Gephi> python3 -m http.server
3. em seguida abra seu navegador e digite na barra de pesquisa http://localhost:8000/
