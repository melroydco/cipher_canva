import { Route, Routes, Navigate } from 'react-router-dom'
import Signup from './app/pages/auth/Signup'
import Signin from './app/pages/auth/Signin'

export function UnAuthenticatedApp () {
  return (
    <div className='bg-[#060505] min-h-screen text-white'>
      <Routes>
        <Route path='/' element={<Signin />} />
        <Route path='/signup' element={<Signup />} />
        <Route path='*' element={<Navigate to='/' />} />
      </Routes>
    </div>
  )
}
