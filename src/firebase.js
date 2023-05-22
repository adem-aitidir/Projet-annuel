import { initializeApp } from 'firebase/app';
import { getFirestore, collection, getDocs } from 'firebase/firestore/lite';


const firebaseConfig = {
    apiKey: "AIzaSyD8ds1ofhDE91hPjFCprFJZBGLNza3a-SE",
    authDomain: "projet-a-652ad.firebaseapp.com",
    projectId: "projet-a-652ad",
    storageBucket: "projet-a-652ad.appspot.com",
    messagingSenderId: "146758137528",
    appId: "1:146758137528:web:b18ef18f41ec7630cee64d"
};

const firebaseapp = initializeApp(firebaseConfig);

const database = getFirestore(firebaseapp);

export { database };