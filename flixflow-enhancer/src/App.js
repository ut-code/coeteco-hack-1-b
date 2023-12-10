import "./App.css";
import React, { useState, useRef } from "react"; 
import { BrowserRouter as Router, Route, Link, useHistory } from "react-router-dom";

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
    <h2 style={{color: "#e74c3c"}}>改善点</h2> 
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
    tipsText: ""
  });
  
  const modelVideoInputRef = useRef(null);
  const studentVideoInputRef = useRef(null); 
  
  const history = useHistory();
  
  const updateFileName = (inputRef, labelId) => {
    // ファイル名を表示するロジック    
  };

  const submitForm = () => {
    // ここにフォーム送信のロジック

    // 送信後にリダイレクト
    history.push("/result");
  };
  
  return (
    <Router>
      <Route path="/" exact>
        <div>
           {/* フォームの実装 */}
        </div>
      </Route>

      <Route path="/result" >  
        <ResultPage />
      </Route>
    </Router>
  );
}

export default App;