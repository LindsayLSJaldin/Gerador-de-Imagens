import mysql from 'mysql2/promise';
import fs from 'fs';
import path from 'path';

const DB = {
  host: '127.0.0.1',
  port: 3306,
  user: 'root',
  password: '11122005',
  database: 'sistemapoliedro',
};

const outputsDir = path.join(process.cwd(), 'outputs');

(async () => {
  try {
    const pool = mysql.createPool(DB);
    await pool.query('DELETE FROM chats');
    console.log('[OK] Chats apagados.');
    if (fs.existsSync(outputsDir)) {
      for (const e of fs.readdirSync(outputsDir)) {
        fs.rmSync(path.join(outputsDir, e), { recursive: true, force: true });
      }
      console.log('[OK] Imagens removidas.');
    } else {
      console.log('[INFO] Pasta outputs não existe.');
    }
    process.exit(0);
  } catch (e) {
    console.error('Falha:', e.message);
    process.exit(1);
  }
})();