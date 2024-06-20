- Instalação no Anaconda:

conda install -c conda-forge mkdocs
conda install -c conda-forge mkdocs-material
conda install -c conda-forge mkdocs-material-extensions
conda install -c conda-forge mkdocstrings
conda install -c conda-forge mkdocstrings-python

https://realpython.com/python-project-documentation-with-mkdocs/

-- Gerar documentação
(venv) $ mkdocs new .


mkdocs build

--- Deploy no github
criar o repositorio no git hub e copiar o endereço

iniciar o git no diretorio root do projeto
git init

adicionar o repositorio criado como remote no Github:

git remote add origin https://github.com/ulf-tecc2/teste_doc.git

adicionar os arquivos ao Github

git add .
git commit -m "Add project code and documentation"
git push origin master