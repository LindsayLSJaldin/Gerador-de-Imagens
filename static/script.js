document.addEventListener("DOMContentLoaded", function () {
  if (window.lucide) lucide.createIcons();

  // Toggle sidebar
  const sidebar = document.getElementById('sidebar');
  const toggleBtn = document.getElementById('toggleSidebar');
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('closed');
      toggleBtn.classList.toggle('sidebar-closed');
      if (window.lucide) lucide.createIcons();
    });
  }

  // Logout button
  const logoutBtn = document.getElementById('logoutBtn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', (e) => {
      e.preventDefault();
      window.location.href = '/login';
    });
  }

  // Ao abrir a página do app, carregue o histórico
  loadChats();
});

// ----------------------------------------------------
// Histórico (mantém como estava)
// ----------------------------------------------------
const subjectsSidebar = [
  'Histórico recente 1', 'Histórico recente 2', 'Histórico recente 3',
  'Histórico recente 4', 'Histórico recente 5'
];
const subjectsTopbar = [
  'Leis de Newton', 'Eletricidade', 'Termologia', 'Óptica',
  'Cinemática', 'Funções'
];

// Prompts prontos por assunto
const subjectPrompts = {
  'Leis de Newton': 'Ilustração educativa de alta qualidade mostrando as três leis de Newton com exemplos visuais: um bloco com força e atrito (1ª lei), dois blocos com setas de força F=ma (2ª lei) e dois patinadores trocando empurrões (3ª lei). Estilo claro, ícones, legendas curtas, fundo branco.',
  'Eletricidade': 'Diagrama limpo de circuito simples com bateria, resistor e lâmpada, mostrando direção da corrente, diferença de potencial e símbolos padrão. Cores suaves, fundo branco, estilo infográfico.',
  'Termologia': 'Infográfico de termologia com condução, convecção e radiação: placas metálicas com gradiente de temperatura, seta de convecção em fluido, radiação do sol. Legendas sucintas, visual minimalista.',
  'Óptica': 'Esquema de óptica com reflexão em espelho plano e refração em prisma, raios de luz com ângulos marcados, índice de refração. Estilo didático, limpo, com cores suaves.',
  'Cinemática': 'Gráfico e ilustrações mostrando MRU e MRUV, com carrinho em trilho, vetores de velocidade e aceleração, tabela pequena com equações. Estilo de apostila.',
  'Funções': 'Funções: informe y=..., f(x)=... ou palavras como linear, quadrática, exponencial. Ex.: y=2x+1; f(x)=x^2-3x+2; seno; log. Intervalo: x in [-5,5].'
};

function getPromptForSubject(s) {
  return subjectPrompts[s] || `Ilustração educativa clara e minimalista sobre "${s}", em estilo de infográfico com legendas curtas, ícones e fundo branco.`;
}

// Render lista lateral fake (se não usar /api/chats)
const subjectList = document.getElementById('subjectList');
if (subjectList) {
  subjectList.innerHTML = '';
  subjectsSidebar.forEach(s => {
    const li = document.createElement('li');
    li.innerHTML = `<button class="subjects-font w-100 text-start px-3 py-2 btn btn-sm">${s}</button>`;
    subjectList.appendChild(li);
  });
}

// Campos de UI usados em vários handlers
const subjectSelect = document.getElementById('subjectSelect');
const wrap = document.getElementById('customSelect');
const btn = document.getElementById('subjectSelectBtn');
const label = document.getElementById('subjectSelectLabel');
const menu = document.getElementById('subjectMenu');

const sendBtn = document.getElementById('sendBtn');
const promptInput = document.getElementById('promptInput');
const chatArea = document.getElementById('chatArea');

// Dropdown custom de assuntos
if (subjectSelect && wrap && btn && label && menu) {
  subjectSelect.querySelectorAll('option:not([value=""])').forEach(o => o.remove());
  subjectsTopbar.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s; opt.textContent = s;
    subjectSelect.appendChild(opt);
  });

  menu.innerHTML = '';
  subjectsTopbar.forEach(s => {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'select-item';
    item.textContent = s;
    item.addEventListener('click', () => {
      subjectSelect.value = s;
      label.textContent = s;
      wrap.classList.remove('open');
      const ready = getPromptForSubject(s);
      if (promptInput) promptInput.value = ready;
    });
    menu.appendChild(item);
  });

  btn.addEventListener('click', () => wrap.classList.toggle('open'));
  document.addEventListener('click', (e) => { if (!wrap.contains(e.target)) wrap.classList.remove('open'); });
}

// ----------------------------------------------------
// Histórico do backend (se existir)
// ----------------------------------------------------
let currentChatId = null;

async function loadChats() {
  try {
    const res = await fetch('/api/chats');
    if (!res.ok) return;
    const chats = await res.json();
    const list = document.getElementById('subjectList');
    if (!list) return;
    list.innerHTML = '';
    chats.forEach(c => {
      const li = document.createElement('li');
      const b = document.createElement('button');
      b.className = 'subjects-font w-100 text-start px-3 py-2 btn btn-sm';
      b.textContent = c.title || 'Sem título';
      b.addEventListener('click', async () => {
        currentChatId = c.id;
        await loadMessages(currentChatId);
      });
      li.appendChild(b);
      list.appendChild(li);
    });
  } catch { /* ignorar */ }
}

async function loadMessages(chatId) {
  try {
    const res = await fetch(`/api/chats/${chatId}/messages`);
    if (!res.ok) return;
    const msgs = await res.json();
    if (!chatArea) return;
    chatArea.innerHTML = '';
    msgs.forEach(m => {
      const rowU = document.createElement('div');
      rowU.className = 'd-flex justify-content-end mb-2';
      const bU = document.createElement('div');
      bU.className = 'bubble bubble-user px-3 py-2';
      bU.textContent = m.prompt;
      rowU.appendChild(bU);
      chatArea.appendChild(rowU);

      const rowB = document.createElement('div');
      rowB.className = 'd-flex justify-content-start mb-2';
      const bB = document.createElement('div');
      bB.className = 'bubble bubble-bot px-3 py-2 wide';
      const url = m.image_url || m.imagem_base64 ? (m.image_url || `data:image/png;base64,${m.imagem_base64}`) : '';
      bB.innerHTML = `
        <div class="mb-2 bot-title">Imagem gerada:</div>
        <div class="chat-media"><div class="chat-image"><img src="${url}" alt="Imagem gerada"></div></div>
        <div class="download-row stick"><a class="download-btn" href="${url}" download="imagem.png">Baixar imagem</a></div>`;
      rowB.appendChild(bB);
      chatArea.appendChild(rowB);
    });
    chatArea.scrollTop = chatArea.scrollHeight;
  } catch { /* ignorar */ }
}

// ----------------------------------------------------
// Heurísticas de geração (Front-end)
// ----------------------------------------------------
const SIMPLE_WORDS = [
  "maçã","maca","apple","dog","cachorro","gato","cat","flor","flower","car","carro","tree","árvore","arvore",
  "montanha","mountain","oceano","ocean","dragão","dragon","castelo","castle","pássaro","passaro","bird",
  "retrato","portrait","personagem","character","copo","glass","mesa","table"
];

const CHART_WORDS = [
  "gráfico","grafico","plot","diagrama","s vs t","s x t","v vs t","v x t","mru","mruv","posição x tempo","posicao x tempo","velocidade x tempo"
];

const FUNCTION_WORDS = [
  "função","funcoes","funções","f(x)","g(x)","y="
];

function decideOptions(raw) {
  const lower = raw.toLowerCase();

  // Gráfico de funções (prioridade máxima)
  const isFunction = FUNCTION_WORDS.some(w => lower.includes(w));
  if (isFunction) {
    return { mode: 'chart_fn' }; // backend vai parsear a expressão a partir do prompt
  }

  // Outros gráficos
  const isChart = CHART_WORDS.some(w => lower.includes(w));
  if (isChart) {
    return { mode: 'chart' };
  }

  // Pedidos fotográficos / simples
  const isSimple = SIMPLE_WORDS.some(w => lower.includes(w)) && lower.length < 220;
  const wantsPhoto = /(foto|fotografia|photo|realista|realistic)/.test(lower);
  const wantsMacro = /(macro)/.test(lower);
  const wantsPaint = /(pintura|painterly|óleo|oleo|oil painting)/.test(lower);
  const wantsDigital = /(arte digital|digital art|concept art|conceitual)/.test(lower);

  let style = 'infographic';
  let ultra = false;
  let precise = false;
  let rawFlag = false;
  let ensemble = 2;

  if (wantsMacro) {
    style = 'macro'; ultra = true; rawFlag = true; ensemble = 3;
  } else if (wantsPhoto || isSimple) {
    style = 'photo'; ultra = true; rawFlag = true; ensemble = 3;
  } else if (wantsPaint) {
    style = 'painterly'; ultra = true; rawFlag = false; ensemble = 2;
  } else if (wantsDigital) {
    style = 'digital_art'; ultra = true; rawFlag = false; ensemble = 2;
  } else {
    // Educacional/descritivo
    style = 'infographic'; precise = true; rawFlag = false; ensemble = 2;
  }

  return { mode: 'ai', style, ultra, precise, raw: rawFlag, ensemble, refiner: true };
}

// ----------------------------------------------------
// Envio de prompt
// ----------------------------------------------------
async function enviarPrompt(e, forcedPrompt) {
  if (e) e.preventDefault();
  if (!promptInput) return;

  const prompt = (forcedPrompt ?? promptInput.value).trim();
  if (!prompt) return;

  // Render prompt do usuário
  if (chatArea) {
    const row = document.createElement('div');
    row.className = 'd-flex justify-content-end mb-2';
    const bubble = document.createElement('div');
    bubble.className = 'bubble bubble-user px-3 py-2';
    bubble.textContent = prompt;
    row.appendChild(bubble);
    chatArea.appendChild(row);
    chatArea.scrollTop = chatArea.scrollHeight;
  }
  if (!forcedPrompt) promptInput.value = '';

  // Placeholder de carregamento
  let placeholderBubble;
  if (chatArea) {
    const botRow = document.createElement('div');
    botRow.className = 'd-flex justify-content-start mb-2';
    placeholderBubble = document.createElement('div');
    placeholderBubble.className = 'bubble bubble-bot px-3 py-2 bubble-loading';
    placeholderBubble.innerHTML = `
      <div class="bot-title d-flex align-items-center gap-2">
        <span>Gerando imagem...</span><span class="spinner" aria-hidden="true"></span>
      </div>`;
    botRow.appendChild(placeholderBubble);
    chatArea.appendChild(botRow);
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  // Monta opções
  const opts = decideOptions(prompt);
  const body = { prompt, ...opts };

  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!res.ok) throw new Error('Falha ao gerar imagem');
    const data = await res.json();

    const dataUrl = data.imagem_base64
      ? `data:image/png;base64,${data.imagem_base64}`
      : (data.image_url || '');

    if (placeholderBubble) {
      placeholderBubble.classList.remove('bubble-loading');
      placeholderBubble.classList.add('wide');
      placeholderBubble.innerHTML = `
        <div class="mb-2 bot-title">Imagem gerada</div>
        <div class="chat-media">
          <div class="chat-image">
            <img src="${dataUrl}" alt="Imagem gerada">
          </div>
        </div>
        <div class="download-row stick">
          <a class="download-btn" href="${dataUrl}" download="imagem.png">Baixar imagem</a>
        </div>`;
      chatArea.scrollTop = chatArea.scrollHeight;
    }

    // Atualiza histórico se o backend retornar chatId
    if (!currentChatId && data.chatId) {
      currentChatId = data.chatId;
      loadChats();
    }
  } catch (err) {
    console.error(err);
    if (placeholderBubble) {
      placeholderBubble.classList.remove('bubble-loading');
      placeholderBubble.textContent = 'Erro ao gerar imagem.';
      chatArea.scrollTop = chatArea.scrollHeight;
    }
  }
}

if (sendBtn) sendBtn.addEventListener('click', enviarPrompt);
if (promptInput) {
  promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      enviarPrompt(e);
    }
  });
}