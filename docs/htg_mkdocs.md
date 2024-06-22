## Instalação no Anaconda:

conda install -c conda-forge mkdocs

conda install -c conda-forge mkdocs-material

conda install -c conda-forge mkdocs-material-extensions

conda install -c conda-forge mkdocstrings

conda install -c conda-forge mkdocstrings-python

pip install mkdocs-bootstrap4

https://realpython.com/python-project-documentation-with-mkdocs/

## Geração da documentação
- Criar estrutura do mkdocs 
(venv) $ mkdocs new .

- Gerar documentação
(venv) $ mkdocs build

- Iniciar servidor
(venv) $ mkdocs serve


## Deploy no github

- criar o repositorio no git hub e copiar o endereço

- iniciar o git no diretorio root do projeto

git init

- adicionar o repositorio criado como remote no Github:

git remote add origin git@github.XXXXXX_ml.git

- adicionar os arquivos ao Github

git add .

git commit -m "Add project code and documentation"

git branch -M main

git push -u origin main