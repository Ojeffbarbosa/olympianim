const caseStudies = [
  {
    id: "mirim-q5",
    title: "OBMEP — Mirim 1 — Q5",
    source: "4ª Olimpíada Mirim 2025 · 1ª fase · Prova Mirim 1 · Questão 5",
    reference: {
      url: "https://olimpiadamirim.obmep.org.br/provas-solucoes",
      label: "Prova e soluções oficiais",
    },
    presentation: {
      video: "assets/examples/mirim-q5-apresentacao.mp4",
    },
    solution: {
      video: "assets/examples/mirim-q5-resolucao.mp4",
    },
    models: [
      ["Planejamento", "Anthropic: claude-opus-4-8"],
      ["Apresentação", "Anthropic: claude-opus-4-8"],
      ["Resolução", "Anthropic: claude-opus-4-8"],
      ["Voz", "Google: gemini-3.1-flash-tts-preview · voz Leda"],
      ["Paleta", "Manim clara — fundo #FFFFFF; primária #236B8E; secundária #699C52; destaque #D55E00"],
    ],
    note: "Os dois vídeos exibidos têm legendas amarelas incorporadas. O projeto também utiliza o objeto visual da formiga enviado pelo professor.",
  },
  {
    id: "nivel-1-q7",
    title: "OBMEP — Nível 1 — Q7",
    source: "20ª OBMEP 2025 · 1ª fase · Nível 1 · Questão 7",
    reference: {
      url: "https://www.obmep.org.br/provas.htm?max-results=7",
      label: "Provas e soluções oficiais",
    },
    presentation: {
      video: "assets/examples/nivel-1-q7-apresentacao.mp4",
    },
    solution: {
      video: "assets/examples/nivel-1-q7-resolucao.mp4",
    },
    models: [
      ["Planejamento", "Google: gemini-3.5-flash"],
      ["Apresentação", "Google: gemini-3.5-flash"],
      ["Resolução", "Google: gemini-3.5-flash"],
      ["Voz", "Google: gemini-3.1-flash-tts-preview · voz Kore"],
      ["Paleta", "Manim escura — fundo #000000; primária #58C4DD; secundária #83C167; destaque #FFFF00"],
    ],
    note: "Edição com IA Gen.: apresentação e resolução foram ajustadas no Assistente Manim após a geração inicial, com auxílio do modelo gpt-5.6-luna da OpenAI.",
  },
  {
    id: "nivel-2-q11",
    title: "OBMEP — Nível 2 — Q11",
    source: "20ª OBMEP 2025 · 1ª fase · Nível 2 · Questão 11",
    reference: {
      url: "https://www.obmep.org.br/provas.htm?max-results=7",
      label: "Provas e soluções oficiais",
    },
    presentation: {
      video: "assets/examples/nivel-2-q11-apresentacao.mp4",
    },
    solution: {
      video: "assets/examples/nivel-2-q11-resolucao.mp4",
    },
    models: [
      ["Planejamento", "OpenAI: gpt-5.6-luna"],
      ["Apresentação", "OpenAI: gpt-5.6-sol"],
      ["Resolução", "OpenAI: gpt-5.6-sol"],
      ["Voz", "Google: gemini-3.1-flash-tts-preview · voz Kore"],
      ["Paleta", "Manim escura — fundo #000000; primária #58C4DD; secundária #83C167; destaque #FFFF00"],
    ],
    note: "Edição com IA Gen.: apresentação e resolução receberam refinamentos solicitados no Assistente Manim com o modelo gpt-5.6-luna da OpenAI.",
  },
  {
    id: "mirim-q6",
    title: "OBMEP — Mirim 1 — Q6",
    source: "4ª Olimpíada Mirim 2025 · 1ª fase · Prova Mirim 1 · Questão 6",
    reference: {
      url: "https://olimpiadamirim.obmep.org.br/provas-solucoes",
      label: "Prova e soluções oficiais",
    },
    presentation: {
      video: "assets/examples/mirim-q6-apresentacao.mp4",
    },
    solution: {
      video: "assets/examples/mirim-q6-resolucao.mp4",
    },
    models: [
      ["Planejamento", "OpenAI: gpt-5.6-luna"],
      ["Apresentação", "OpenAI: gpt-5.6-luna"],
      ["Resolução", "OpenAI: gpt-5.6-luna"],
      ["Voz", "Google: gemini-3.1-flash-tts-preview · voz Kore"],
      ["Paleta", "Manim escura — fundo #000000; primária #58C4DD; secundária #83C167; destaque #FFFF00"],
    ],
    note: "Edição com IA Gen.: a apresentação foi ajustada com gpt-5.6-luna; a resolução foi refinada com gpt-5.6-luna e alguns ajustes manuais.",
  },
  {
    id: "nivel-2-q1",
    title: "OBMEP — Nível 2 — Q1",
    source: "20ª OBMEP 2025 · 1ª fase · Nível 2 · Questão 1",
    reference: {
      url: "https://www.obmep.org.br/provas.htm?max-results=7",
      label: "Provas e soluções oficiais",
    },
    presentation: {
      video: "assets/examples/nivel-2-q1-apresentacao.mp4",
    },
    solution: {
      video: "assets/examples/nivel-2-q1-resolucao.mp4",
    },
    models: [
      ["Planejamento", "Google: gemini-3.5-flash"],
      ["Apresentação", "Google: gemini-3.5-flash"],
      ["Resolução", "Google: gemini-3.5-flash"],
      ["Voz", "Google: gemini-3.1-flash-tts-preview · voz Zephyr"],
      ["Paleta", "Manim clara — fundo #FFFFFF; primária #236B8E; secundária #699C52; destaque #D55E00"],
    ],
    note: "Planejamento, apresentação e resolução foram produzidos com o mesmo modelo Google; a narração utilizou uma voz Google independente.",
  },
  {
    id: "nivel-3-q1",
    title: "OBMEP — Nível 3 — Q1",
    source: "20ª OBMEP 2025 · 1ª fase · Nível 3 · Questão 1",
    reference: {
      url: "https://www.obmep.org.br/provas.htm?max-results=7",
      label: "Provas e soluções oficiais",
    },
    presentation: {
      video: "assets/examples/nivel-3-q1-apresentacao.mp4",
    },
    solution: {
      video: "assets/examples/nivel-3-q1-resolucao.mp4",
    },
    models: [
      ["Planejamento", "Anthropic: claude-sonnet-5"],
      ["Apresentação", "Anthropic: claude-opus-4-8"],
      ["Resolução", "Anthropic: claude-opus-4-8"],
      ["Voz", "Google: gemini-3.1-flash-tts-preview · voz Kore"],
      ["Paleta", "Manim escura — fundo #000000; primária #58C4DD; secundária #83C167; destaque #FFFF00"],
    ],
    note: "Edição com IA Gen.: apresentação e resolução foram ajustadas no Assistente Manim com os próprios modelos da Anthropic.",
  },
  {
    id: "nivel-3-q8",
    title: "OBMEP — Nível 3 — Q8",
    source: "20ª OBMEP 2025 · 1ª fase · Nível 3 · Questão 8",
    reference: {
      url: "https://www.obmep.org.br/provas.htm?max-results=7",
      label: "Provas e soluções oficiais",
    },
    presentation: {
      video: "assets/examples/nivel-3-q8-apresentacao.mp4",
    },
    solution: {
      video: "assets/examples/nivel-3-q8-resolucao.mp4",
    },
    models: [
      ["Planejamento", "Anthropic: claude-sonnet-5"],
      ["Apresentação", "Anthropic: claude-sonnet-5"],
      ["Resolução", "Anthropic: claude-sonnet-5"],
      ["Voz", "Google: gemini-3.1-flash-tts-preview · voz Kore"],
      ["Paleta", "Manim escura — fundo #000000; primária #58C4DD; secundária #83C167; destaque #FFFF00"],
    ],
    note: "O mesmo modelo Anthropic participou do planejamento e da construção dos dois vídeos; a voz foi sintetizada pelo Google.",
  },
  {
    id: "nivel-3-q15",
    title: "OBMEP — Nível 3 — Q15",
    source: "20ª OBMEP 2025 · 1ª fase · Nível 3 · Questão 15",
    reference: {
      url: "https://www.obmep.org.br/provas.htm?max-results=7",
      label: "Provas e soluções oficiais",
    },
    presentation: {
      video: "assets/examples/nivel-3-q15-apresentacao.mp4",
    },
    solution: {
      video: "assets/examples/nivel-3-q15-resolucao.mp4",
    },
    models: [
      ["Planejamento", "Google: gemini-3.1-flash-lite"],
      ["Apresentação", "Google: gemini-3.1-flash-lite e gemini-3.1-pro-preview"],
      ["Resolução", "Google: gemini-3.1-flash-lite"],
      ["Voz", "Google: gemini-2.5-flash-preview-tts · voz Orus"],
      ["Paleta", "Manim clara — fundo #FFFFFF; primária #236B8E; secundária #699C52; destaque #D55E00"],
    ],
    note: "Edição com IA Gen.: a apresentação foi refinada no Assistente Manim com o modelo gemini-3.5-flash.",
  },
  {
    id: "elon-q1",
    title: "Competição Elon Lages Lima — Q1",
    source: "VII Competição Elon Lages Lima de Matemática 2026 · Questão 1",
    reference: {
      url: "https://www.obm.org.br/content/uploads/2026/06/7_Competicao_Elon_Lages_Lima_2026_Gabarito.pdf",
      label: "Caderno e gabarito oficiais (PDF)",
      secondaryUrl: "https://www.obm.org.br/competicao-elon-lages-lima-de-matematica/",
      secondaryLabel: "Página da competição",
    },
    presentation: {
      video: "assets/examples/elon-q1-apresentacao.mp4",
    },
    solution: {
      video: "assets/examples/elon-q1-resolucao.mp4",
    },
    models: [
      ["Planejamento", "Google: gemini-3.5-flash"],
      ["Apresentação", "Google: gemini-3.5-flash"],
      ["Resolução", "Google: gemini-3.5-flash"],
      ["Voz", "Google: gemini-3.1-flash-tts-preview · voz Iapetus"],
      ["Paleta", "Manim clara — fundo #FFFFFF; primária #236B8E; secundária #699C52; destaque #D55E00"],
    ],
    note: "Edição com IA Gen.: a apresentação foi ajustada no Assistente Manim com o modelo gpt-5.6-luna. Os dois vídeos exibidos têm legendas amarelas incorporadas.",
  },
  {
    id: "elon-q2",
    title: "Competição Elon Lages Lima — Q2",
    source: "VII Competição Elon Lages Lima de Matemática 2026 · Questão 2",
    reference: {
      url: "https://www.obm.org.br/content/uploads/2026/06/7_Competicao_Elon_Lages_Lima_2026_Gabarito.pdf",
      label: "Caderno e gabarito oficiais (PDF)",
      secondaryUrl: "https://www.obm.org.br/competicao-elon-lages-lima-de-matematica/",
      secondaryLabel: "Página da competição",
    },
    presentation: {
      video: "assets/examples/elon-q2-apresentacao.mp4",
    },
    solution: {
      video: "assets/examples/elon-q2-resolucao.mp4",
    },
    models: [
      ["Planejamento", "OpenAI: gpt-5.6-luna"],
      ["Apresentação", "OpenAI: gpt-5.6-sol"],
      ["Resolução", "OpenAI: gpt-5.6-sol"],
      ["Voz", "Google: gemini-3.1-flash-tts-preview · voz Kore"],
      ["Paleta", "Manim escura — fundo #000000; primária #58C4DD; secundária #83C167; destaque #FFFF00"],
    ],
    note: "Edição com IA Gen.: a resolução foi ajustada no Assistente Manim e também passou pelo corretor técnico automático.",
  },
];

const caseStudyOrder = [
  "nivel-2-q1",
  "nivel-1-q7",
  "mirim-q5",
  "elon-q1",
  "nivel-2-q11",
  "nivel-3-q1",
  "nivel-3-q8",
  "mirim-q6",
  "nivel-3-q15",
  "elon-q2",
];

caseStudies.sort(
  (left, right) => caseStudyOrder.indexOf(left.id) - caseStudyOrder.indexOf(right.id),
);

const tourScreens = [
  {
    src: "assets/screenshots/01-inicio-configuracao.png",
    alt: "Tela inicial do Olympianim com opções carregadas de IA Gen, voz e paleta",
    caption: "Configure provedor, modelo, chave da sessão, voz, idioma, velocidade e paleta antes de criar o projeto.",
  },
  {
    src: "assets/screenshots/02-projetos-carregados.png",
    alt: "Tela com a lista carregada de projetos salvos no Olympianim",
    caption: "Consulte projetos reais, seus estados e modelos; reabra ou exporte cada registro.",
  },
  {
    src: "assets/screenshots/03-producao-carregada.png",
    alt: "Tela de produção com as sete etapas concluídas e controles carregados",
    caption: "Acompanhe base matemática, planos, códigos e renderizações e selecione o modelo da próxima chamada.",
  },
  {
    src: "assets/screenshots/03b-producao-video.png",
    alt: "Vídeo de apresentação carregado na tela de produção",
    caption: "Assista à apresentação gerada antes de alternar para a resolução do mesmo projeto.",
  },
  {
    src: "assets/screenshots/03c-producao-resolucao.png",
    alt: "Vídeo de resolução carregado na tela de produção",
    caption: "Alterne para a resolução e confira o desenvolvimento matemático renderizado.",
  },
  {
    src: "assets/screenshots/04-editor-carregado.png",
    alt: "Editor de código Manim com controle de versões e assistente de IA Gen",
    caption: "Edite Python com realce de sintaxe e linhas numeradas, alterne o vídeo e use o assistente de IA Gen.",
  },
  {
    src: "assets/screenshots/05-consumo-carregado.png",
    alt: "Painel de consumo com métricas, tabela e gráficos carregados",
    caption: "Acompanhe chamadas concluídas, falhas, custos e distribuição por provedor com dados reais.",
  },
  {
    src: "assets/screenshots/06-prompts-carregados.png",
    alt: "Gerenciador de prompts com agente, variáveis, versão e template carregados",
    caption: "Consulte variáveis, edite templates, crie versões e preserve o prompt utilizado em cada projeto.",
  },
  {
    src: "assets/screenshots/07-config-geral.png",
    alt: "Configurações gerais de entrega, qualidade e Assistente Manim",
    caption: "Defina o formato de entrega, a qualidade de renderização e o modelo padrão do Assistente Manim.",
  },
  {
    src: "assets/screenshots/08-config-modelos-ia.png",
    alt: "Catálogo de modelos de IA Gen com provedores, preços e estados",
    caption: "Gerencie modelos de OpenAI, Google e Anthropic, incluindo estado, ordem e preços de referência.",
  },
  {
    src: "assets/screenshots/09-config-modelos-voz.png",
    alt: "Catálogo de modelos de voz com preços e configurações",
    caption: "Gerencie modelos de voz, provedor, unidade de cobrança e modelo padrão.",
  },
  {
    src: "assets/screenshots/10-config-paletas.png",
    alt: "Configuração de paletas visuais do Olympianim",
    caption: "Visualize as cores de cada paleta, duplique um padrão e crie variações para as animações.",
  },
];

const examplesGrid = document.querySelector("[data-examples-grid]");

function modelRows(rows) {
  return rows
    .map(
      ([role, model]) =>
        `<div class="model-row"><dt>${role}</dt><dd>${model}</dd></div>`,
    )
    .join("");
}

function renderCaseStudies() {
  if (!examplesGrid) return;

  examplesGrid.innerHTML = caseStudies
    .map(
      (item, index) => `
        <article class="example-card reveal" data-case-id="${item.id}">
          <header class="example-header">
            <span class="example-sequence">ESTUDO ${String(index + 1).padStart(2, "0")}</span>
            <div>
              <h3 class="example-title">${item.title}</h3>
              <p class="example-source"><strong>Fonte:</strong> ${item.source}</p>
            </div>
            <div class="reference-links">
              <a href="${item.reference.url}" target="_blank" rel="noopener noreferrer">${item.reference.label} ↗</a>
              ${item.reference.secondaryUrl ? `<a href="${item.reference.secondaryUrl}" target="_blank" rel="noopener noreferrer">${item.reference.secondaryLabel} ↗</a>` : ""}
            </div>
          </header>
          <div class="case-videos">
            <div class="generated-output">
              <p class="panel-label"><span>01</span> vídeos gerados pelo protótipo</p>
              <div class="example-type-tabs" role="group" aria-label="Escolher entre o vídeo de apresentação e o vídeo de resolução">
                <button class="is-active" type="button" data-kind="presentation" aria-pressed="true">
                  <span>01</span>
                  <span class="type-copy"><strong>Apresentação</strong><small>introduz o problema</small></span>
                </button>
                <button type="button" data-kind="solution" aria-pressed="false">
                  <span>02</span>
                  <span class="type-copy"><strong>Resolução</strong><small>desenvolve a solução</small></span>
                </button>
              </div>
              <div class="example-media">
                <video
                  controls
                  playsinline
                  preload="metadata"
                  src="${item.presentation.video}"
                  aria-label="Apresentação do caso ${item.title}"
                ></video>
              </div>
            </div>
          </div>
          <div class="example-meta">
            <div>
              <p class="panel-label"><span>02</span> registro de produção</p>
              <dl class="model-ledger">${modelRows(item.models)}</dl>
            </div>
            <p class="case-note">${item.note}</p>
          </div>
        </article>`,
    )
    .join("");

  examplesGrid.querySelectorAll(".example-card").forEach((card) => {
    const item = caseStudies.find((entry) => entry.id === card.dataset.caseId);
    const video = card.querySelector("video");
    const buttons = card.querySelectorAll("[data-kind]");
    if (!item || !video) return;

    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        const kind = button.dataset.kind;
        const media = item[kind];
        if (!media) return;
        video.pause();
        video.src = media.video;
        video.setAttribute(
          "aria-label",
          `${kind === "solution" ? "Resolução" : "Apresentação"} do caso ${item.title}`,
        );
        video.load();
        buttons.forEach((candidate) => {
          const selected = candidate === button;
          candidate.classList.toggle("is-active", selected);
          candidate.setAttribute("aria-pressed", String(selected));
        });
      });
    });

  });
}

function setupTour() {
  const tabs = document.querySelectorAll("[data-tour-index]");
  const image = document.querySelector("[data-tour-image]");
  const caption = document.querySelector("[data-tour-caption]");
  if (!image || !caption) return;

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const screen = tourScreens[Number(tab.dataset.tourIndex)];
      if (!screen) return;
      tabs.forEach((candidate) => {
        candidate.setAttribute("aria-selected", String(candidate === tab));
      });
      image.animate([{ opacity: 0.25 }, { opacity: 1 }], { duration: 280 });
      image.src = screen.src;
      image.alt = screen.alt;
      caption.textContent = screen.caption;
    });
  });
}

function setupMenu() {
  const toggle = document.querySelector("[data-menu-toggle]");
  const nav = document.querySelector("[data-nav]");
  if (!toggle || !nav) return;

  let menuScrollPosition = 0;

  const close = () => {
    const wasOpen = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-label", "Abrir menu");
    nav.classList.remove("is-open");
    document.body.classList.remove("menu-open");
    if (wasOpen) window.scrollTo(0, menuScrollPosition);
  };

  toggle.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") !== "true";
    if (!open) {
      close();
      return;
    }
    menuScrollPosition = window.scrollY;
    toggle.setAttribute("aria-expanded", "true");
    toggle.setAttribute("aria-label", "Fechar menu");
    nav.classList.add("is-open");
    document.body.classList.add("menu-open");
  });
  nav.querySelectorAll("a").forEach((link) => link.addEventListener("click", close));
  window.addEventListener("resize", () => {
    if (window.innerWidth > 840) close();
  });
}

function setupReveal() {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const elements = document.querySelectorAll(".reveal");
  if (reducedMotion || !("IntersectionObserver" in window)) {
    elements.forEach((element) => element.classList.add("is-visible"));
    return;
  }
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.12 },
  );
  elements.forEach((element) => observer.observe(element));
}

function setupHeader() {
  const header = document.querySelector("[data-header]");
  const update = () => header?.classList.toggle("is-scrolled", window.scrollY > 18);
  update();
  window.addEventListener("scroll", update, { passive: true });
}

function pauseHiddenVideos() {
  if (!("IntersectionObserver" in window)) return;
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting && !entry.target.paused) entry.target.pause();
      });
    },
    { threshold: 0.05 },
  );
  document.querySelectorAll(".example-card video").forEach((video) => observer.observe(video));
}

renderCaseStudies();
setupTour();
setupMenu();
setupReveal();
setupHeader();
pauseHiddenVideos();
