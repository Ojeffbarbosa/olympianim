# Política de segurança

## Versões com suporte

As correções de segurança são aplicadas à versão mais recente disponível no repositório.

## Como comunicar uma vulnerabilidade

Não abra uma issue pública que contenha credenciais, dados privados de projetos ou detalhes que
permitam explorar uma vulnerabilidade. Utilize um aviso de segurança privado do GitHub
(*Security Advisory*) para entrar em contato com o responsável pelo repositório.

Se uma chave de provedor tiver sido exposta, revogue-a imediatamente no painel do respectivo
serviço. Não aguarde a análise do relato para substituir a credencial.

Ao relatar uma vulnerabilidade, informe, quando possível:

- a versão ou o commit utilizado;
- o comportamento observado e o comportamento esperado;
- os passos mínimos para reprodução;
- o impacto estimado;
- uma proposta de correção, se houver.

Remova chaves, dados pessoais, caminhos locais e conteúdo privado antes de anexar registros ou
arquivos.

## Limites de segurança

- Chaves carregadas de `.env` ou digitadas no Streamlit não são gravadas nos registros dos
  projetos.
- Para executar as funções solicitadas, o aplicativo pode enviar ao provedor de IA selecionado
  enunciados, instruções, soluções fornecidas pelo professor, imagens anexadas, planos, trechos de
  código e mensagens do Assistente Manim. O provedor de voz recebe os textos destinados à
  síntese de narração.
- Se o tracing opcional do LangSmith for habilitado, informações das execuções dos agentes podem
  ser transmitidas ao LangSmith. Esse recurso permanece desativado por padrão.
- O tratamento, a retenção e a eventual utilização dos dados transmitidos são regidos pelos
  termos e pelas políticas dos provedores externos. Não processe dados pessoais, sigilosos ou
  materiais sem autorização para compartilhamento.
- O código Manim gerado passa por verificações baseadas em uma lista restrita de construções AST
  e é renderizado em um processo separado, com cancelamento, tempo limite e uma lista permitida
  de variáveis de ambiente.
- Segredos de provedores são removidos do ambiente de renderização. Somente a credencial do
  serviço de voz selecionado é fornecida quando a narração está habilitada.
- Exportações de auditoria omitem SQLite, arquivos de ambiente, logs brutos de renderização,
  caminhos da máquina e formatos reconhecidos de chaves. Imagens e vídeos originais só são
  incluídos após confirmação explícita dos direitos de uso.
- Cópias do SQLite anteriores a migrações podem conter conteúdo privado. Mantenha
  `workspace/.backups/` local e protegido com as mesmas permissões do banco principal.
- As verificações AST reduzem usos acidentais indevidos, mas **não constituem uma sandbox do
  sistema operacional**.
- Execute o Olympianim localmente com uma conta sem privilégios administrativos.
- Revise o código gerado antes de aprovar a renderização e não processe projetos de origem não
  confiável.

## Escopo

Relatos sobre indisponibilidade, preços, limites ou comportamento dos provedores externos devem
ser encaminhados aos respectivos provedores. Erros funcionais que não envolvam segurança podem
ser registrados em uma issue pública, desde que não exponham informações sensíveis.
