import { initializeApp } from "https://www.gstatic.com/firebasejs/10.9.0/firebase-app.js";
import { getAuth, 
         GoogleAuthProvider } from "https://www.gstatic.com/firebasejs/10.9.0/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/10.9.0/firebase-firestore.js";

// Your web app's Firebase configuration
const firebaseConfig = {
    'apiKey': "AIzaSyCtUoSBmt-cLOujXKjWlNB4AbrpiumqRO0",
    'authDomain': "kesl-56091.firebaseapp.com",
    'projectId': "kesl-56091",
    'storageBucket': "kesl-56091.firebasestorage.app",
    'messagingSenderId': "110569187589",
    'appId': "1:110569187589:web:135de78bd50d656a3f2f6c",
    'measurementId': "G-BVWN0ZHGSP"
};

  // Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

const db = getFirestore(app);

export { auth, provider, db };