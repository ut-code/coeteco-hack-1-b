import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Link } from 'react-router-dom';

const ImprovementSection = ({ number, teacherImage, studentImage, description }) => (
  <section className="improvement-section">
    <h3>改善点{number}</h3>
    <div className="image-container">
      <img src={teacherImage} alt={`教師の画像${number}`} />
      <img src={studentImage} alt={`生徒の画像${number}`} />
    </div>
    <p>{description}</p>
  </section>
);

const ResultPage = () => (
  <div className="result-container">
    <h2 style={{ color: '#e74c3c' }}>改善点</h2>

    <ImprovementSection
      number={1}
      teacherImage="teacher_image_1.jpg"
      studentImage="student_image_1.jpg"
      description="このフレームでは、姿勢が安定していません。生徒さんはもう少し背筋を伸ばすように心がけましょう。"
    />

    <ImprovementSection
      number={2}
      teacherImage="teacher_image_2.jpg"
      studentImage="student_image_2.jpg"
      description="手の位置が不安定です。教師のように手をしっかり支えて行うと良いでしょう。"
    />

    {/* 同様に他の改善点を追加 */}

    <button className="back-to-index">
      <Link to="/">Index.html に戻る</Link>
    </button>
  </div>
);

const App = () => {
  const [formData, setFormData] = useState({
    modelVideo: null,
    studentVideo: null,
    tipsText: '',
  });

  const updateFileName = (input, labelId) => {
    const label = document.getElementById(labelId);
    if (input.files.length > 0) {
      label.textContent = input.files[0].name;
    } else {
      label.textContent = 'Choose File';
    }
  };

  const submitForm = () => {
    // フォーム送信のロジックを実装

    // フォーム送信が成功した場合
    setFormData({
      modelVideo: null,
      studentVideo: null,
      tipsText: '',
    });
  };

  return (
    <Router>
      <Route path="/" exact>
        <div>
          <header>
            <div className="header-content">
              <h1>FlipFlow-Enhancer</h1>
              <p>小学生低学年の前転を改善するツール</p>
            </div>
          </header>

          <main>
            <form id="uploadForm">
              {/* フォームの内容をここに記述 */}
              <button type="button" onClick={submitForm}>
                提出
              </button>
            </form>
          </main>
        </div>
      </Route>

      <Route path="/result">
        <ResultPage />
      </Route>
    </Router>
  );
};

export default App;
