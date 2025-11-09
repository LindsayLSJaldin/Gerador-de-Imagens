document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('loginForm');
  const emailInput = document.getElementById('email');
  const passwordInput = document.getElementById('password');
  const btn = document.getElementById('btnLogin');
  const errBox = document.getElementById('loginError');
  const ALLOWED_DOMAIN = 'sistemapoliedro.com.br';

  if (!form || !btn) return;

  const showError = (msg) => {
    if (!errBox) return;
    errBox.textContent = msg;
    errBox.classList.add('show');
  };
  const clearError = () => {
    if (!errBox) return;
    errBox.classList.remove('show');
    errBox.textContent = '';
  };

  form.addEventListener('submit', (e) => e.preventDefault());

  btn.addEventListener('click', async () => {
    clearError();

    const email = (emailInput?.value || '').trim();
    const password = (passwordInput?.value || '').trim();

    if (!email || !password) {
      showError('Seu email ou senha estão incorretos. Tente novamente.');
      return;
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      showError('Seu email ou senha estão incorretos. Tente novamente.');
      return;
    }
    const domain = email.split('@').pop().toLowerCase();
    if (domain !== ALLOWED_DOMAIN) {
      showError('Seu email ou senha estão incorretos. Tente novamente.');
      return;
    }

    try {
      btn.disabled = true;
      btn.textContent = 'Entrando...';

      const res = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data?.ok) {
        // mensagem pedida
        showError('Seu email ou senha estão incorretos. Tente novamente.');
        return;
      }

      window.location.href = '/app';
    } catch {
      showError('Falha ao conectar ao servidor.');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Logar';
    }
  });

  // Limpa erro ao digitar
  [emailInput, passwordInput].forEach(i => i?.addEventListener('input', clearError));
});