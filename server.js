const express = require('express');
const fetch = require('node-fetch');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

const DATA_FILE = path.join(__dirname, 'data', 'history.json');
const API_URL = 'https://era-technology-particular-domestic.trycloudflare.com/api/tx';

// ─── Data persistence ─────────────────────────────────────────────────────────
function loadHistory() {
  try {
    if (fs.existsSync(DATA_FILE)) {
      const raw = fs.readFileSync(DATA_FILE, 'utf8');
      return JSON.parse(raw);
    }
  } catch (e) {}
  return { sessions: {}, ordered: [] };
}

function saveHistory(data) {
  try {
    fs.mkdirSync(path.dirname(DATA_FILE), { recursive: true });
    fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2));
  } catch (e) {
    console.error('Save error:', e.message);
  }
}

let historyData = loadHistory();

// ─── Prediction Algorithms ────────────────────────────────────────────────────

/**
 * 1. Markov Chain (order 3): analyze transition probability T->T, T->X, X->T, X->X
 *    using the last 3 outcomes as state
 */
function markovPredict(results) {
  if (results.length < 4) return null;
  const order = Math.min(3, Math.floor(results.length / 3));
  const transitions = {};

  for (let i = order; i < results.length; i++) {
    const state = results.slice(i - order, i).join('');
    const next = results[i];
    if (!transitions[state]) transitions[state] = { T: 0, X: 0 };
    transitions[state][next]++;
  }

  const currentState = results.slice(-order).join('');
  const trans = transitions[currentState];
  if (!trans || (trans.T === 0 && trans.X === 0)) return null;

  const total = trans.T + trans.X;
  const probT = trans.T / total;
  return {
    prediction: probT >= 0.5 ? 'T' : 'X',
    confidence: Math.max(probT, 1 - probT),
    method: 'Markov Chain',
    detail: `T:${trans.T} X:${trans.X} (state="${currentState}")`
  };
}

/**
 * 2. Pattern Frequency: find recurring sub-patterns of length 2-5
 *    and predict next based on what historically follows
 */
function patternPredict(results) {
  if (results.length < 6) return null;
  let bestScore = 0;
  let bestPred = null;
  let bestDetail = '';

  for (let len = 5; len >= 2; len--) {
    const pattern = results.slice(-len).join('');
    const counts = { T: 0, X: 0 };

    for (let i = len; i < results.length; i++) {
      const seg = results.slice(i - len, i).join('');
      if (seg === pattern) {
        counts[results[i]]++;
      }
    }

    const total = counts.T + counts.X;
    if (total >= 3) {
      const score = Math.max(counts.T, counts.X) / total;
      if (score > bestScore) {
        bestScore = score;
        bestPred = counts.T >= counts.X ? 'T' : 'X';
        bestDetail = `pattern="${pattern}" T:${counts.T} X:${counts.X} (${total} matches)`;
      }
    }
  }

  if (!bestPred) return null;
  return {
    prediction: bestPred,
    confidence: bestScore,
    method: 'Pattern Frequency',
    detail: bestDetail
  };
}

/**
 * 3. Streak / Anti-streak: detect long streaks and apply regression-to-mean
 *    Also detect alternating patterns
 */
function streakPredict(results) {
  if (results.length < 3) return null;

  // Check alternating pattern
  let isAlternating = true;
  for (let i = results.length - 4; i < results.length - 1; i++) {
    if (i >= 0 && results[i] === results[i + 1]) { isAlternating = false; break; }
  }
  if (isAlternating && results.length >= 4) {
    const next = results[results.length - 1] === 'T' ? 'X' : 'T';
    return { prediction: next, confidence: 0.75, method: 'Alternating Pattern', detail: 'detected TXTX... pattern' };
  }

  // Streak detection
  let streak = 1;
  const last = results[results.length - 1];
  for (let i = results.length - 2; i >= 0 && results[i] === last; i--) streak++;

  if (streak >= 4) {
    const opp = last === 'T' ? 'X' : 'T';
    const conf = Math.min(0.55 + streak * 0.05, 0.85);
    return { prediction: opp, confidence: conf, method: 'Streak Break', detail: `streak of ${streak} "${last}", expecting reversal` };
  }

  if (streak >= 2) {
    return { prediction: last, confidence: 0.55, method: 'Streak Continue', detail: `streak of ${streak} "${last}", continuing` };
  }

  return null;
}

/**
 * 4. Weighted Moving Average on dice sums
 */
function diceWMAPredict(sessions) {
  const withSums = sessions.filter(s => s.tong && s.tong > 0);
  if (withSums.length < 5) return null;

  const recent = withSums.slice(-10);
  let weightedSum = 0, totalWeight = 0;
  recent.forEach((s, i) => {
    const w = i + 1;
    weightedSum += s.tong * w;
    totalWeight += w;
  });
  const wma = weightedSum / totalWeight;
  // Tài = sum 11-18, Xỉu = sum 3-10, boundary ~10.5
  const prediction = wma > 10.5 ? 'T' : 'X';
  const distance = Math.abs(wma - 10.5);
  const confidence = Math.min(0.5 + distance / 14, 0.85);

  return {
    prediction,
    confidence,
    method: 'Dice WMA',
    detail: `WMA sum = ${wma.toFixed(2)} (threshold 10.5)`
  };
}

/**
 * 5. Bayesian ensemble: combine all methods with weighted voting
 */
function ensemblePredict(results, sessions) {
  const methods = [
    { weight: 0.35, result: markovPredict(results) },
    { weight: 0.30, result: patternPredict(results) },
    { weight: 0.20, result: streakPredict(results) },
    { weight: 0.15, result: diceWMAPredict(sessions) }
  ];

  const votes = { T: 0, X: 0 };
  const usedMethods = [];

  for (const { weight, result } of methods) {
    if (result) {
      const score = weight * result.confidence;
      votes[result.prediction] += score;
      usedMethods.push({ ...result, weight });
    }
  }

  const total = votes.T + votes.X;
  if (total === 0) {
    // fallback: global frequency
    const tCount = results.filter(r => r === 'T').length;
    const xCount = results.filter(r => r === 'X').length;
    return {
      prediction: tCount >= xCount ? 'T' : 'X',
      confidence: Math.max(tCount, xCount) / (tCount + xCount),
      methods: [],
      votes
    };
  }

  const pred = votes.T >= votes.X ? 'T' : 'X';
  const conf = Math.max(votes.T, votes.X) / total;

  return { prediction: pred, confidence: conf, methods: usedMethods, votes };
}

// ─── Stats helper ─────────────────────────────────────────────────────────────
function computeStats(sessions) {
  const results = sessions.map(s => s.ket_qua).filter(r => r === 'T' || r === 'X');
  const tCount = results.filter(r => r === 'T').length;
  const xCount = results.filter(r => r === 'X').length;

  // Accuracy of our past predictions
  const withPred = sessions.filter(s => s.our_prediction && s.ket_qua);
  const correct = withPred.filter(s => s.our_prediction === s.ket_qua).length;
  const accuracy = withPred.length > 0 ? (correct / withPred.length * 100).toFixed(1) : null;

  return { total: sessions.length, tCount, xCount, accuracy, correctPredictions: correct, totalPredictions: withPred.length };
}

// ─── API Routes ───────────────────────────────────────────────────────────────

// Fetch from source API and process
app.get('/api/fetch', async (req, res) => {
  try {
    const response = await fetch(API_URL, { timeout: 10000 });
    if (!response.ok) throw new Error(`API responded ${response.status}`);
    const data = await response.json();

    const phien = data.phien_hien_tai || data.phien_truoc;
    if (!phien) return res.json({ success: false, error: 'No session ID in response', raw: data });

    const history = loadHistory();
    let isNew = false;

    // Save completed previous session
    if (data.phien_truoc && data.ket_qua && !history.sessions[data.phien_truoc]) {
      const ketQua = data.ket_qua.toLowerCase().includes('tài') || data.ket_qua === 'T' ? 'T'
        : data.ket_qua.toLowerCase().includes('xỉu') || data.ket_qua === 'X' ? 'X' : null;

      if (ketQua) {
        history.sessions[data.phien_truoc] = {
          phien: data.phien_truoc,
          xuc_xac1: data.xuc_xac1,
          xuc_xac2: data.xuc_xac2,
          xuc_xac3: data.xuc_xac3,
          tong: data.tong,
          ket_qua: ketQua,
          ket_qua_raw: data.ket_qua,
          our_prediction: history.sessions[data.phien_truoc]?.our_prediction || null,
          timestamp: Date.now()
        };
        if (!history.ordered.includes(data.phien_truoc)) {
          history.ordered.push(data.phien_truoc);
          isNew = true;
        }
      }
    }

    // Generate prediction for current session
    const orderedSessions = history.ordered.map(id => history.sessions[id]).filter(Boolean);
    const results = orderedSessions.map(s => s.ket_qua).filter(r => r === 'T' || r === 'X');
    const ensemble = ensemblePredict(results, orderedSessions);

    // Save our prediction for the current session
    const currentPhien = data.phien_hien_tai;
    if (currentPhien && !history.sessions[currentPhien]) {
      history.sessions[currentPhien] = {
        phien: currentPhien,
        our_prediction: ensemble.prediction,
        confidence: ensemble.confidence,
        status: 'pending',
        timestamp: Date.now()
      };
      if (!history.ordered.includes(currentPhien)) {
        history.ordered.push(currentPhien);
      }
    } else if (currentPhien && history.sessions[currentPhien] && !history.sessions[currentPhien].our_prediction) {
      history.sessions[currentPhien].our_prediction = ensemble.prediction;
      history.sessions[currentPhien].confidence = ensemble.confidence;
    }

    saveHistory(history);
    historyData = history;

    const stats = computeStats(orderedSessions);

    res.json({
      success: true,
      raw: data,
      prediction: ensemble,
      stats,
      isNew,
      history: history.ordered.slice(-50).reverse().map(id => history.sessions[id]).filter(Boolean)
    });
  } catch (e) {
    res.json({ success: false, error: e.message });
  }
});

// Get current history
app.get('/api/history', (req, res) => {
  const history = loadHistory();
  const ordered = history.ordered.map(id => history.sessions[id]).filter(Boolean);
  const stats = computeStats(ordered);
  res.json({
    history: ordered.slice(-100).reverse(),
    stats,
    total: ordered.length
  });
});

// Clear history
app.delete('/api/history', (req, res) => {
  historyData = { sessions: {}, ordered: [] };
  saveHistory(historyData);
  res.json({ success: true });
});

// Health check for Render
app.get('/health', (req, res) => res.json({ status: 'ok', time: new Date().toISOString() }));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🎲 Tài Xỉu Predictor running on port ${PORT}`));
