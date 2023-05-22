import React, { useEffect, useState } from 'react';
import './Stats.css';
import axios from "axios";
import StatsRow from './StatsRow';
import { database } from './firebase';


const url = "https://finnhub.io/api/v1/quote";
const token = "ch7a5thr01qt83gc5it0ch7a5thr01qt83gc5itg";

function Stats() {
    const [stockData, setStockData] = useState([])
    const [myStock, setmyStock] = useState([])
    const getMyStocks = () => {
        database
            .collection('mystocks')
            .onSnapshot(snapshot => {
                console.log(snapshot)
            })
    }

    const getStockData = (stock) => {
        return axios
            .get(`${url}?symbol=${stock}&token=${token}`)
            .catch((error) => {
                console.error("Error", error.message);
            });
    };
    useEffect(() => {
        let respData = []
        const stockList = ["AAPL", "MSFT", "TSLA", "BABA", "UBER", "DIS", "SBUX"];

        let promises = [];
        //getMyStocks();
        stockList.map((stock) => {
            promises.push(
                getStockData(stock)
                .then((res) => {
                    respData.push({
                        name: stock,
                        ...res.data
                    });
                })
            )
        });

        Promise.all(promises).then(() => {
            setStockData(respData)
        })

    }, [])


    return ( <
        div className = 'stats' >
        <
        div className = 'stats__container' >
        <
        div className = 'stats__header' >
        <
        p > Stocks < /p> <
        /div> <
        div className = 'stats__content' >
        <
        div className = 'stats__rows' >

        <
        /div> <
        /div> <
        div className = 'stats__header' >
        <
        p > Lists < /p> <
        /div> <
        div className = 'stats__content' >
        <
        div className = 'stats__rows' > {
            stockData.map((stock) => ( <
                StatsRow key = { stock.name }
                name = { stock.name }
                openPrice = { stock.o }
                price = { stock.c }
                />
            ))
        }

        <
        /div> <
        /div>

        <
        /div>

        <
        /div>
    )
}

export default Stats