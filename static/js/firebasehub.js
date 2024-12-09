// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";  // Import Firebase Authentication
import { getDatabase } from "firebase/database"; // Import Firebase Realtime Database

// Your web app's Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyCaCzq-Pqibf-fkz1SI7thA66D7k8MXPZo",
    authDomain: "lu-smart-cf7e5.firebaseapp.com",
    projectId: "lu-smart-cf7e5",
    storageBucket: "lu-smart-cf7e5.firebasestorage.app",
    messagingSenderId: "783021291028",
    appId: "1:783021291028:web:4c678aded7de8cdc299901"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication and Realtime Database
const auth = getAuth(app);
const database = getDatabase(app);

// Export the Firebase services for use in other parts of your app
export { auth, database };
