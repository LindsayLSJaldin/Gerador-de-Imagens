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
});

const subjectsSidebar = [
  'Histórico recente 1', 'Histórico recente 2', 'Histórico recente 3',
  'Histórico recente 4', 'Histórico recente 5'
];
const subjectsTopbar = [
  'Leis de Newton', 'Eletricidade', 'Termologia', 'Óptica', 
  'Grandezas Proporcionais', 'Probabilidade','Dinâmica',
  'Geometria Espacial', 'Análise Combinatória', 'Cinemática', 
  'Estática', 'Gravitação', 'Termodinâmica', 'Ondas',
  'Reflexão e Refração', 'Interferência e Difração', 'Funções',
  'Trigonometria', 'Geometria Analítica', 'Geometria Plana',
  'Progressões Aritméticas e Geométricas', 'Conjuntos',
  'Matrizes e Determinantes', 'Polinômios', 'Números Complexos'
];

// Lista lateral
const subjectList = document.getElementById('subjectList');
if (subjectList) {
  subjectList.innerHTML = '';
  subjectsSidebar.forEach(s => {
    const li = document.createElement('li');
    li.innerHTML = `<button class="subjects-font w-100 text-start px-3 py-2 btn btn-sm">${s}</button>`;
    subjectList.appendChild(li);
  });
}

// Select nativo (oculto) + dropdown custom
const subjectSelect = document.getElementById('subjectSelect');
const wrap = document.getElementById('customSelect');
const btn = document.getElementById('subjectSelectBtn');
const label = document.getElementById('subjectSelectLabel');
const menu = document.getElementById('subjectMenu');

if (subjectSelect && wrap && btn && label && menu) {
  // limpa e repopula select oculto
  subjectSelect.querySelectorAll('option:not([value=""])').forEach(o => o.remove());
  subjectsTopbar.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s; opt.textContent = s;
    subjectSelect.appendChild(opt);
  });

  // cria itens do dropdown custom
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
    });
    menu.appendChild(item);
  });

  // abre/fecha menu
  btn.addEventListener('click', () => {
    wrap.classList.toggle('open');
  });

  // fecha ao clicar fora
  document.addEventListener('click', (e) => {
    if (!wrap.contains(e.target)) wrap.classList.remove('open');
  });
}

// Envio prompt
const sendBtn = document.getElementById('sendBtn');
const promptInput = document.getElementById('promptInput');
const chatArea = document.getElementById('chatArea');

async function enviarPrompt(e) {
  if (e) e.preventDefault();
  if (!promptInput) return;

  const prompt = promptInput.value.trim();
  if (!prompt) return;

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

  promptInput.value = '';

  let placeholderBubble;
  if (chatArea) {
    const botRow = document.createElement('div');
    botRow.className = 'd-flex justify-content-start mb-2';

    placeholderBubble = document.createElement('div');
    placeholderBubble.className = 'bubble bubble-bot px-3 py-2 bubble-loading';
    placeholderBubble.innerHTML = `
      <div class="bot-title d-flex align-items-center gap-2">
        <span>Gerando imagem...</span>
        <span class="spinner" aria-hidden="true"></span>
      </div>
    `;
    botRow.appendChild(placeholderBubble);
    chatArea.appendChild(botRow);
    chatArea.scrollTop = chatArea.scrollHeight;
  }
  // chamada à API (quando pronto)
  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    if (!res.ok) throw new Error('Falha ao gerar imagem');
    const data = await res.json();
    const dataUrl = `data:image/png;base64,${data.imagem_base64}`;

    if (placeholderBubble) {
      placeholderBubble.classList.remove('bubble-loading');
      // garante bolha mais larga para o botão ir além da imagem
      placeholderBubble.classList.add('wide');

      placeholderBubble.innerHTML = `
        <div class="mb-2 bot-title">Aqui está a sua imagem:</div>
        <div class="chat-media">
          <div class="chat-image">
            <img src="${dataUrl}" alt="Imagem gerada">
          </div>
        </div>
        <!-- botão fora do .chat-media (desacoplado) -->
        <div class="download-row stick">
          <a class="download-btn" href="${dataUrl}" download="imagem.png">Baixar imagem</a>
        </div>
      `;
      chatArea.scrollTop = chatArea.scrollHeight;
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
    // Shift+Enter = nova linha; Enter = enviar
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      enviarPrompt(e);
    }
  });
}