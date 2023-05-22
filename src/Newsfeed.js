import React from 'react'
import './Newsfeed.css';
import LineGraph from './LineGraph';
import TimeLine from './TimeLine';

function Newsfeed() {
    return ( <
        div className = 'newsfeed' >
        <
        div className = 'newsfeed__container' >
        <
        div className = 'newsfeed__chartSection' >
        <
        div className = 'newsfeed__portofolio' >
        <
        h1 > 114, 656€ < /h1> <
        p > +44.63€(+0.04 % ) Today < /p>

        <
        /div> <
        /div> <
        div className = 'newsfeed__chart' >
        <
        LineGraph / >
        <
        TimeLine / >

        <
        /div>

        <
        /div> <
        div className = 'newsfeed__buying__section' >
        <
        h2 > Buying power < /h2> <
        h2 > $4 .11 < /h2>

        <
        /div>

        <
        /div>
    )
}

export default Newsfeed