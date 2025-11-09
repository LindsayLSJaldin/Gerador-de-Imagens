import fs from 'fs';
import path from 'path';
import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'url';
import axios from 'axios';
import http from 'http';
import https from 'https';
import dotenv from 'dotenv';
import mysql from 'mysql2/promise';
import bcrypt from 'bcrypt';
import crypto from 'crypto';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Carrega .env (opcional, se existir)
dotenv.config({ path: path.join(__dirname, '.env') });

// Config DB manual (edite aqui)
const DB_CONFIG = {
  host: '127.0.0.1',
  port: 3306,
  user: 'root',
  password: '11122005',
  database: 'sistemapoliedro',
};

// Pool MySQL usando .env OU fallback manual
const pool = mysql.createPool({
  host: process.env.DB_HOST || DB_CONFIG.host,
  port: Number(process.env.DB_PORT || DB_CONFIG.port),
  user: process.env.DB_USER || DB_CONFIG.user,
  password: process.env.DB_PASS || DB_CONFIG.password,
  database: process.env.DB_NAME || DB_CONFIG.database,
  waitForConnections: true,
  connectionLimit: 10,
});

// Log da config efetiva (sem senha)
console.log('[DB] Usando config:', {
  host: process.env.DB_HOST || DB_CONFIG.host,
  port: Number(process.env.DB_PORT || DB_CONFIG.port),
  user: process.env.DB_USER || DB_CONFIG.user,
  database: process.env.DB_NAME || DB_CONFIG.database,
});

// Teste rápido ao iniciar
(async () => {
  try {
    await pool.query('SELECT 1');
    console.log('[DB] Conectado');
  } catch (e) {
    console.error('[DB] Falha na conexão:', e.message);
  }
})();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

const STATIC_DIR = path.join(__dirname, 'static');
const TEMPLATES_DIR = path.join(__dirname, 'templates');
const OUTPUTS_DIR = path.join(__dirname, 'outputs'); // onde salvaremos as imagens
fs.mkdirSync(OUTPUTS_DIR, { recursive: true });

app.use('/static', express.static(STATIC_DIR));
app.use('/outputs', express.static(OUTPUTS_DIR)); // servir as imagens geradas
app.get('/', (_req, res) => res.sendFile(path.join(TEMPLATES_DIR, 'login.html')));
app.get('/login', (_req, res) => res.sendFile(path.join(TEMPLATES_DIR, 'login.html')));
app.get('/forgot', (_req, res) => res.sendFile(path.join(TEMPLATES_DIR, 'forgot.html')));
app.get('/app', (_req, res) => res.sendFile(path.join(TEMPLATES_DIR, 'index.html')));

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

async function initDb() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS chats (
      id CHAR(36) PRIMARY KEY,
      title VARCHAR(120) NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
  `);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS messages (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      chat_id CHAR(36) NOT NULL,
      prompt TEXT NOT NULL,
      image_url VARCHAR(255) NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      INDEX (chat_id),
      CONSTRAINT fk_messages_chats FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
    );
  `);
}

function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }

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

// Rota de login (com domínio restrito)
app.post('/api/login', async (req, res) => {
  try {
    const { email, password } = req.body || {};
    if (!email || !password) return res.status(400).json({ error: 'Campos obrigatórios' });

    const domain = String(email).split('@').pop()?.toLowerCase();
    if (domain !== 'sistemapoliedro.com.br') {
      return res.status(401).json({ error: 'Domínio de e-mail não autorizado' });
    }

    const [rows] = await pool.execute(
      'SELECT id, name, password_hash FROM users WHERE email = ? LIMIT 1',
      [email]
    );
    if (!rows.length) return res.status(401).json({ error: 'Credenciais inválidas' });

    const user = rows[0];
    const ok = await bcrypt.compare(password, user.password_hash);
    if (!ok) return res.status(401).json({ error: 'Credenciais inválidas' });

    return res.json({ ok: true, user: { id: user.id, name: user.name, email } });
  } catch (e) {
    console.error('Login error:', e);
    return res.status(500).json({ error: 'Falha ao autenticar' });
  }
});

app.post('/api/forgot', async (req, res) => {
  try {
    const { email } = req.body || {};
    if (!email) return res.status(400).json({ error: 'E-mail obrigatório' });
    const domain = String(email).split('@').pop()?.toLowerCase();
    if (domain !== 'sistemapoliedro.com.br') {
      return res.status(401).json({ error: 'Domínio não autorizado' });
    }
    // Verifica se existe
    const [rows] = await pool.execute(
      'SELECT id FROM users WHERE email = ? LIMIT 1',
      [email]
    );
    if (!rows.length) {
      // Para segurança, responde genérico
      return res.json({ ok: true, message: 'Se existir conta, enviaremos instruções.' });
    }
    // Aqui você implementaria envio de e-mail / token
    console.log('[FORGOT] Solicitação de redefinição para', email);
    return res.json({ ok: true, message: 'Verifique seu e-mail e redefina sua senha.' });
  } catch (e) {
    console.error('Forgot error:', e);
    return res.status(500).json({ error: 'Falha ao processar' });
  }
});

app.post('/generate', async (req, res) => {
  const { prompt, width, height, steps, guidance, chatId } = req.body || {};
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

    // cria/resolve chat
    let cid = chatId;
    if (!cid) {
      cid = crypto.randomUUID();
      const title = (prompt.length > 60 ? prompt.slice(0, 57) + '...' : prompt);
      await pool.execute('INSERT INTO chats (id, title) VALUES (?, ?)', [cid, title]);
    }

    // salva a imagem em disco
    const chatDir = path.join(OUTPUTS_DIR, cid);
    ensureDir(chatDir);
    const filename = `${Date.now()}.png`;
    const filePath = path.join(chatDir, filename);
    fs.writeFileSync(filePath, Buffer.from(b64, 'base64'));
    const imageUrl = `/outputs/${cid}/${filename}`;

    // registra a mensagem
    await pool.execute(
      'INSERT INTO messages (chat_id, prompt, image_url) VALUES (?, ?, ?)',
      [cid, prompt, imageUrl]
    );

    console.log('[Proxy] OK, base64 len:', b64.length, '->', imageUrl);
    return res.json({ imagem_base64: b64, chatId: cid, imageUrl });
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

// Histórico: lista chats
app.get('/api/chats', async (_req, res) => {
  const [rows] = await pool.query(`
    SELECT c.id, c.title, c.created_at,
      (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS count
    FROM chats c
    ORDER BY c.created_at DESC
  `);
  res.json(rows);
});

// Histórico: mensagens de um chat
app.get('/api/chats/:id/messages', async (req, res) => {
  const [rows] = await pool.execute(
    'SELECT id, prompt, image_url, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC',
    [req.params.id]
  );
  res.json(rows);
});

// Inicia o servidor só após aguardar o Python e DB
async function start() {
  await initDb();
  await waitForPython();
  const server = app.listen(PORT, () => {
    console.log(`Server on http://localhost:${PORT}`);
  });
  server.headersTimeout = 0;
  server.requestTimeout = 0;
  server.keepAliveTimeout = 120000;
}
start();