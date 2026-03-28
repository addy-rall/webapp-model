// ─────────────────────────────────────────────────────────────
//  The Sentiment Herald — Express.js Proxy Server
//  Forwards requests to HuggingFace Inference API with CORS
// ─────────────────────────────────────────────────────────────

const express = require('express');
const cors    = require('cors');
const fetch   = (...args) => import('node-fetch').then(({ default: f }) => f(...args));
const path    = require('path');

const app = express();

// ── Config ──────────────────────────────────────────────────
const HF_API_KEY = process.env.HF_API_KEY;
const HF_MODEL   = 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis';

const PORT       = 'process.env.PORT || 3001';

// ── Middleware ───────────────────────────────────────────────
app.use(cors());                        
app.use(express.json());               
app.use(express.static(path.join(__dirname, 'public'))); // Serve HTML from /public

// ── Health check ─────────────────────────────────────────────
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', model: HF_MODEL });
});

// ── Sentiment proxy endpoint ──────────────────────────────────
app.post('/api/sentiment', async (req, res) => {
  const { text } = req.body;

  if (!text || typeof text !== 'string' || text.trim().length < 3) {
    return res.status(400).json({ error: 'Invalid or missing "text" field.' });
  }

  console.log(`[${new Date().toISOString()}] Analysing: "${text.slice(0, 80)}..."`);

  try {
    const hfRes = await fetch(
      `https://router.huggingface.co/hf-inference/models/${HF_MODEL}`,
      {
        method:  'POST',
        headers: {
          'Content-Type':  'application/json',
          'Authorization': `Bearer ${HF_API_KEY}`
        },
        body: JSON.stringify({ inputs: text })
      }
    );

    // Model cold-starting on HuggingFace free tier
    if (hfRes.status === 503) {
      const errBody = await hfRes.json().catch(() => ({}));
      return res.status(503).json({
        error:             'Model is loading, please retry in ~20 seconds.',
        estimated_time:    errBody.estimated_time || 20
      });
    }

    if (!hfRes.ok) {
      const errBody = await hfRes.text().catch(() => '');
      return res.status(hfRes.status).json({ error: errBody || `HuggingFace error ${hfRes.status}` });
    }

    const data = await hfRes.json();
    // HF returns [[{label,score},...]] — unwrap one level
    const scores = Array.isArray(data[0]) ? data[0] : data;
    res.json({ scores });

  } catch (err) {
    console.error('Proxy error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── Start ─────────────────────────────────────────────────────
app.listen(PORT, () => {
  
  console.log(`    Local:   http://localhost:${PORT}`);
  console.log(`    Health:  http://localhost:${PORT}/api/health`);
  console.log(`    Model:   ${HF_MODEL}\n`);
});
