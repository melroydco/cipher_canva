import { Route, Routes } from 'react-router-dom'
import { Home } from './app/pages/main/Home'
import { Encode } from './app/pages/main/Encode'
import { Decode } from './app/pages/main/Decode'
import { About } from './app/pages/main/About'
import { EncodedSuccess } from './app/pages/main/EncodedSuccess'
import { DecodedSuccess } from './app/pages/main/DecodedSuccess'
import { Main } from './app/pages/main/Main'

export function AuthenticatedApp () {
  return (
    <div className='bg-[#060505] min-h-screen text-white'>
      <Routes>
        <Route element={<Main />}>
          <Route index element={<Home />} />
          <Route path='encode' index element={<Encode />} />
          <Route path='decode' index element={<Decode />} />
          <Route path='about' element={<About />} />
          <Route path='encode/success' element={<EncodedSuccess />} />
          <Route path='decode/success' element={<DecodedSuccess />} />
        </Route>
      </Routes>
    </div>
  )
}
