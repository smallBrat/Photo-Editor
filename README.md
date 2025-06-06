## 📄 `README.md`

````markdown
# 📸 Smart Photo Editor with Gemini AI

A Streamlit web app that lets you edit images and generate Instagram-style captions or song recommendations using Google's Gemini AI.

---

## ✨ Features

- 🎛️ Brightness, resize, black & white, highlight, and text overlay edits
- 🎨 Mood filters: Sunny, Cool, Warm, Dreamy, Moody
- 📝 Generate AI captions with Gemini
- 🎵 Get song suggestions based on image vibe
- 🚀 Simple and free to deploy with Streamlit Cloud

---

## 🖼️ Demo

Try it live: [your-deployment-link](https://your-username.streamlit.app)

---

## 📦 Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run app.py
````

---

## 🔐 Setup API Key (Gemini AI)

1. Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a `.streamlit/secrets.toml` file:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

---

## 📁 Project Structure

```
├── app.py                # Main Streamlit app
├── requirements.txt      # Dependencies
└── .streamlit/
    └── secrets.toml      # API key (not committed)
```

---

## 📜 License

MIT License

---

## 🙋‍♂️ Author

Made with ❤️ by [Your Name](https://github.com/your-username)
