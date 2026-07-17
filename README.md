# Olympianim

[![Qualidade](https://github.com/Ojeffbarbosa/olympianim/actions/workflows/quality.yml/badge.svg)](https://github.com/Ojeffbarbosa/olympianim/actions/workflows/quality.yml)

O Olympianim é um protótipo local de autoria docente para criar vídeos de apresentação e
resolução de problemas olímpicos de Matemática com IA generativa, Manim e narração opcional.
O sistema mantém o professor no processo de decisão: planos e códigos são revisados antes da
renderização e podem ser editados, regenerados e versionados.

Página do projeto: <https://ojeffbarbosa.github.io/olympianim/>

Este projeto foi desenvolvido como parte da pesquisa intitulada “Sistema multiagente com
supervisão humana para a criação de animações multimodais de problemas olímpicos com Manim e IA
generativa”.

## O que o sistema produz

1. **Apresentação do problema:** organiza e apresenta o enunciado sem revelar a resposta.
2. **Resolução comentada:** desenvolve a solução em etapas após a tentativa dos estudantes.

Também é possível combinar os vídeos, gerar arquivos SRT, incorporar ou remover legendas do MP4,
editar o código Manim manualmente ou por chat e exportar um projeto para auditoria.

## Como funciona

O fluxo é dividido em duas etapas relacionadas. Quando necessário, um solucionador produz ou
interpreta uma base matemática, que precisa ser aprovada pelo usuário. Essa base orienta a
apresentação sem expor a resposta e é reutilizada posteriormente na resolução.

Em cada vídeo, um planejador propõe a sequência didática e um agente construtor gera o código
Manim, a narração e a sincronização. O usuário aprova separadamente plano e código. Se a
renderização falhar por um erro técnico, um corretor pode tentar reparar o código por até três
vezes. Depois da geração, o código continua disponível para edições manuais ou assistidas por IA.

## Tecnologias principais

- **Streamlit:** interface local de autoria.
- **LangGraph:** fluxo persistente e interrupções *human-in-the-loop*.
- **LangChain:** integração dos agentes com OpenAI, Google e Anthropic.
- **Manim Community e Manim Voiceover:** animação, narração e sincronização.
- **SQLite:** projetos, checkpoints, trabalhos, prompts, versões, eventos e consumo.
- **Pydantic:** validação dos contratos estruturados entre as etapas.

## Requisitos

- Python 3.12;
- FFmpeg;
- Cairo e Pango;
- LaTeX, `dvisvgm` e fontes recomendadas.

No Ubuntu 24.04:

```bash
sudo apt-get update
sudo apt-get install --yes \
  build-essential dvisvgm ffmpeg libcairo2-dev libpango1.0-dev pkg-config \
  texlive-fonts-recommended texlive-latex-base texlive-latex-extra texlive-science
```

Para outros sistemas operacionais, consulte as instruções de instalação do Manim Community.

## Instalação

Com `uv` e o arquivo de lock versionado:

```bash
uv sync --all-extras --dev --locked
```

Alternativa com `pip`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[all]"
```

## Credenciais de API

Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

Para gerar conteúdo, é necessária uma chave de pelo menos um provedor de IA compatível:
OpenAI, Google ou Anthropic. Para narração, é necessária uma credencial da OpenAI ou do Google.
Esses serviços são externos ao projeto e podem cobrar pelo uso segundo seus próprios preços e
termos. Como o código é aberto, outras integrações podem ser implementadas.

Preencha apenas as variáveis dos provedores que pretende utilizar. As opções aceitas estão
descritas em `.env.example`. Chaves digitadas na interface permanecem na memória do processo; o
banco registra apenas a origem da credencial. Nunca publique `.env`, logs privados ou projetos
com dados sensíveis.

Ao utilizar uma API externa, o conteúdo necessário à operação pode ser enviado ao provedor
selecionado. Isso pode incluir enunciados, instruções, soluções fornecidas pelo professor, imagens
anexadas, planos, trechos de código, mensagens do Assistente Manim e textos de narração. Serviços
de voz recebem o texto que será sintetizado. Se o tracing opcional do LangSmith for habilitado,
dados das execuções dos agentes também podem ser enviados a esse serviço. O tratamento e a
retenção desses dados seguem os termos e as políticas de cada provedor. Não envie informações
pessoais, sigilosas ou materiais que você não esteja autorizado a compartilhar.

## Execução

```bash
uv run streamlit run src/olympianim/app.py
```

Ou, com o ambiente virtual ativo:

```bash
streamlit run src/olympianim/app.py
```

Quando instalado como pacote, o comando abaixo inicia a mesma interface:

```bash
olympianim
```

Por padrão, `.env` e `workspace/` são lidos a partir do diretório atual. Defina
`OLYMPIANIM_HOME` para escolher outro diretório de dados.

## Persistência e execução em segundo plano

As ações são persistidas no SQLite e executadas por um worker local. Cada decisão é vinculada à
fase correspondente e a um identificador idempotente. Uma resposta concluída da IA pode ser
reaproveitada após uma queda, evitando repetir a chamada. A geração continua se o navegador for
recarregado ou fechado, desde que o processo Streamlit permaneça ativo.


## Estrutura do repositório

```text
.github/                # integração contínua e publicação do site
page/                   # site estático do projeto
src/olympianim/
├── app.py              # navegação Streamlit
├── pages/              # páginas da interface
├── database/           # esquema e repositório SQLite
├── graph/              # fluxo LangGraph
├── manim/              # renderização, voz e proteção de execução
├── prompts/            # modelos de prompt editáveis e versionados
├── providers/llm/      # integrações OpenAI, Google e Anthropic
├── schemas/            # contratos Pydantic
├── services/           # casos de uso e worker local
├── tools/              # ferramentas disponibilizadas aos agentes
└── ui/                 # componentes e estado da interface
tests/                  # testes automatizados
typings/                # stubs locais usados pela análise estática
```

Os artefatos locais são armazenados em `workspace/projects/<project_id>/` e não são versionados.

## Solução de problemas

### Manim não encontrado

Execute os comandos na mesma virtualenv usada pelo Streamlit e confirme:

```bash
uv run python -c "import manim; print(manim.__version__)"
```

### Erro de LaTeX ou `dvisvgm`

Confirme `latex --version` e `dvisvgm --version` e instale os pacotes indicados em **Requisitos**.

### FFmpeg não encontrado

Confirme `ffmpeg -version`. O Manim precisa encontrar o executável no `PATH`.

### Trabalho interrompido ou sem credencial

Abra novamente o projeto. Trabalhos sem atividade são recuperados; se a chave existia apenas na
sessão, informe-a novamente e use **Repetir etapa** ou **Retomar etapa**.

### Falha após três correções

Abra o Editor Manim, revise o diagnóstico e renderize uma nova versão manualmente.

## Limitações conhecidas

- O projeto é um protótipo e pode apresentar falhas.
- As saídas matemáticas e pedagógicas precisam de revisão docente.
- Modelos e APIs externas podem mudar, ficar indisponíveis ou gerar custos.
- A qualidade visual varia conforme o problema, o material fornecido e o modelo escolhido.
- A consulta à documentação do Manim depende de acesso à internet.
- O worker é local e único, não distribuído.
- A renderização possui cancelamento cooperativo e timeout de 15 minutos.
- As verificações do código reduzem usos acidentais indevidos, mas não constituem uma sandbox de
  sistema operacional.
- Os estudos de caso atuais não demonstram efeito sobre a aprendizagem.

## Segurança e privacidade

Leia [SECURITY.md](SECURITY.md) antes de processar materiais de terceiros ou publicar uma
instalação. Não inclua credenciais, bancos locais, logs ou dados privados em issues e exportações.

## Materiais de terceiros

A licença MIT se aplica exclusivamente ao código-fonte original do aplicativo Olympianim. Ela
não abrange enunciados de provas, recortes de questões, marcas, imagens, vídeos demonstrativos
derivados desses materiais nem qualquer outro conteúdo pertencente a terceiros. Esses materiais
permanecem sujeitos aos direitos e às condições definidos por seus respectivos titulares e não
são relicenciados pela licença do aplicativo.

As fontes e as condições aplicáveis aos vídeos dos estudos de caso estão detalhadas no
[aviso da pasta de exemplos](page/assets/examples/README.md).

## Citação e licença

Os metadados para citação estão em [CITATION.cff](CITATION.cff). Somente o código-fonte original
do aplicativo é distribuído sob a licença MIT; consulte [LICENSE](LICENSE).
