import { useEffect, useState } from 'react'
import { Link, NavLink, Outlet } from 'react-router-dom'
import { useAuth } from '../../auth-context'
import clsx from 'clsx'

function getNavLinkClassName ({ isActive }) {
  return clsx('font-bold', isActive ? 'title-text-gradient' : '')
}

export const Main = () => {
  const { setUser, user } = useAuth()

  const signOut = () => {
    localStorage.removeItem('USER')
    setUser(null)
  }
  return (
    <div className='min-h-screen flex flex-col'>
      <div className='flex items-center justify-between gap-12 px-12 py-6'>
        <Link to='/'>
          <h1 className='text-xl'>Welcome {user.name}!</h1>
        </Link>
        <nav className='flex items-center justify-center gap-12 flex-1 p-6 bg-[#222] rounded-full'>
          <NavLink className={getNavLinkClassName} to=''>
            Home
          </NavLink>
          <NavLink className={getNavLinkClassName} to='encode'>
            Encode
          </NavLink>
          <NavLink className={getNavLinkClassName} to='decode'>
            Decode
          </NavLink>
          <NavLink className={getNavLinkClassName} to='about'>
            About us
          </NavLink>
        </nav>
        <button onClick={signOut}>Logout</button>
      </div>
      <Outlet />
    </div>
  )
}
