// 必要なモジュールをインストール
// npm install express

// サーバーサイドコード (server.jsとして保存)
const express = require('express');
const multer = require('multer'); // ファイルアップロードを処理するためのミドルウェア

const app = express();
const port = 5000;

// ファイルアップロードの設定
const upload = multer({ dest: 'uploads/' });

// 静的ファイルの提供 (例: HTML, CSSなど)
app.use(express.static('public'));

// POSTリクエストを処理するエンドポイント
app.post('/upload', upload.fields([{ name: 'modelVideo' }, { name: 'studentVideo' }]), (req, res) => {
  // 送信されたデータをコンソールに表示
  console.log('Received data:', req.body);
  console.log('Uploaded files:', req.files);

  // ここでデータの処理や応答を行う

  // クライアントにレスポンスを送信
  res.json({ message: 'Data received successfully.' });
});

// サーバーを起動
app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
