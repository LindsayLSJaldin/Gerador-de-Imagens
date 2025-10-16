import fs from 'fs';
import path from 'path';
import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'url';
import axios from 'axios';
import http from 'http';
import https from 'https';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Diretórios estáticos e página
const STATIC_DIR = path.join(__dirname, 'static');
const TEMPLATES_DIR = path.join(__dirname, 'templates');
app.use('/static', express.static(STATIC_DIR));
app.get('/', (_req, res) => res.sendFile(path.join(TEMPLATES_DIR, 'index.html')));

// Aumenta timeouts do Express/Node para requisições longas (CPU)
app.use((req, res, next) => {
  req.setTimeout(0);
  res.setTimeout(0);
  next();
});

// Cliente Axios para o serviço Python
const PY_URL = process.env.GEN_URL || 'http://127.0.0.1:5001/generate';
const PY_HEALTH = 'http://127.0.0.1:5001/health';
const genClient = axios.create({
  timeout: 1800000,
  maxContentLength: Infinity,
  maxBodyLength: Infinity,
  httpAgent: new http.Agent({ keepAlive: true }),
  httpsAgent: new https.Agent({ keepAlive: true }),
});

async function waitForPython(maxTries = 30) {
  for (let i = 1; i <= maxTries; i++) {
    try {
      await axios.get(PY_HEALTH, { timeout: 5000 });
      console.log('[Proxy] Python OK');
      return;
    } catch {
      console.log(`[Proxy] Aguardando Python... tent.${i}/${maxTries}`);
      await new Promise(r => setTimeout(r, 2000));
    }
  }
  console.warn('[Proxy] Python não respondeu ao /health. Continuando mesmo assim.');
}

async function callWithRetry(fn, tries = 3, delayMs = 1500) {
  try {
    return await fn();
  } catch (e) {
    const retryable = e?.code === 'ECONNREFUSED' || e?.code === 'ETIMEDOUT';
    if (tries > 0 && retryable) {
      console.warn(`[Proxy] retry em ${delayMs}ms por ${e.code}...`);
      await new Promise(r => setTimeout(r, delayMs));
      return callWithRetry(fn, tries - 1, delayMs * 2);
    }
    throw e;
  }
}

app.get('/health', (_req, res) => res.json({ ok: true }));

app.post('/generate', async (req, res) => {
  const { prompt, width, height, steps, guidance } = req.body || {};
  if (!prompt || !String(prompt).trim()) {
    return res.status(400).json({ error: 'Prompt vazio' });
  }
  try {
    const payload = {
      prompt,
      width: width ?? 512,
      height: height ?? 512,
      steps: steps ?? 30,
      guidance: guidance ?? 7.0,
    };
    console.log('[Proxy] requisitando ao Python:', payload);

    const { data } = await callWithRetry(() => genClient.post(PY_URL, payload));

    const b64 = data?.imagem_base64;
    if (!b64 || typeof b64 !== 'string' || b64.length < 1000) {
      console.error('[Proxy] Resposta inválida do Python:', data);
      return res.status(502).json({ error: 'Resposta inválida do serviço Python' });
    }
    console.log('[Proxy] OK, base64 len:', b64.length);
    return res.json({ imagem_base64: b64 });
  } catch (err) {
    const code = err?.code || '';
    const msg = err?.response?.data || err.message;
    console.error('Generate proxy error:', code, msg);
    if (code === 'ECONNREFUSED' || code === 'ETIMEDOUT') {
      return res.status(502).json({ error: 'Serviço de geração indisponível. Inicie o Python em 127.0.0.1:5001.' });
    }
    if (err?.response?.data?.error) {
      return res.status(500).json({ error: err.response.data.error });
    }
    return res.status(500).json({ error: 'Falha na geração' });
  }
});

// Inicia o servidor só após aguardar o Python
async function start() {
  await waitForPython();
  const server = app.listen(PORT, () => {
    console.log(`Server on http://localhost:${PORT}`);
  });
  server.headersTimeout = 0;
  server.requestTimeout = 0;
  server.keepAliveTimeout = 120000;
}
start();