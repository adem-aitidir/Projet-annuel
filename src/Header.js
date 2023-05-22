import React from 'react'
import Logo from './yoda_img.svg'
import './Header.css'


function Header() {
    return ( <
        div className = 'header__wrapper' > { /*logo*/ } <
        div className = 'header__logo' >
        <
        img src = { Logo }
        width = { 95 }
        /> <
        /div> { /*Search */ } <
        div className = 'header__search' >
        <
        div className = 'header__searchContainer' >
        <
        input placeholder = 'Search'
        type = 'text' / >
        <
        /div>

        <
        /div> <
        div className = 'header__menuItems' >
        <
        a href = "#" > Free Stocks < /a> <
        a href = "#" > Portofolio < /a> <
        a href = "#" > Cash < /a> <
        a href = "#" > Messages < /a> <
        a href = "#" > Account < /a> <
        /div>

        <
        /div>

    )
}

export default Header