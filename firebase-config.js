/*
============================================================
FIREBASE CONFIGURATION

Shared by every page (index, login, signup, dashboard).
REPLACE THESE VALUES WITH YOUR ACTUAL FIREBASE CONFIG.
============================================================
*/

import { initializeApp }
    from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";

import { getAuth }
    from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

  const firebaseConfig = {
    apiKey: "AIzaSyD2Ctr1g32K84SAcuYnJ_EgJvrxHBD3UsI",
    authDomain: "offgrid-3b579.firebaseapp.com",
    projectId: "offgrid-3b579",
    storageBucket: "offgrid-3b579.firebasestorage.app",
    messagingSenderId: "520512562949",
    appId: "1:520512562949:web:5e9d8f485019a4723b796c",
    measurementId: "G-ZZJGMWY0PT"
  };


const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
