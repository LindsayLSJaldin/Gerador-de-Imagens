document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('forgotForm');
  const emailInput = document.getElementById('forgotEmail');
  const btn = document.getElementById('btnForgot');
  const errBox = document.getElementById('forgotError');
  const okBox = document.getElementById('forgotSuccess');
  const ALLOWED_DOMAIN = 'sistemapoliedro.com.br';

  const show = (el, msg) => {
    el.textContent = msg;
    el.classList.add('show');
  };
  const hideAll = () => {
    [errBox, okBox].forEach(b => { b.textContent=''; b.classList.remove('show'); });
  };

  if (form) form.addEventListener('submit', e => e.preventDefault());

  btn?.addEventListener('click', async () => {
    hideAll();
    const email = (emailInput?.value || '').trim();
    if (!email) { show(errBox, 'Informe o e-mail.'); return; }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) { show(errBox, 'E-mail inválido.'); return; }

    const domain = email.split('@').pop().toLowerCase();
    if (domain !== ALLOWED_DOMAIN) { show(errBox, 'Domínio não autorizado.'); return; }

    try {
      btn.disabled = true;
      btn.textContent = 'Enviando...';
      const res = await fetch('/api/forgot', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ email })
      });
      const data = await res.json().catch(()=>({}));
      if (!res.ok || !data?.ok) {
        show(errBox, data?.error || 'Falha ao processar.');
        return;
      }
      show(okBox, data.message || 'Verifique sua caixa de entrada.');
    } catch {
      show(errBox, 'Erro de conexão.');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Redefinir senha';
    }
  });

  emailInput?.addEventListener('input', hideAll);
});