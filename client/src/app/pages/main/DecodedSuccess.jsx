import React from 'react'
import { useSearchParams } from 'react-router-dom'

export const DecodedSuccess = () => {
  const [searchParams] = useSearchParams()
  const text = searchParams.get('text')
  return (
    <div className='flex items-center flex-col gap-4'>
      <h1 className='title-text-gradient text-center text-6xl'>Decoder</h1>
      <p className='text-center'>
        Successfully decoded the hidden message within the encoded image,
        revealing its concealed text.
      </p>
      <div className='bg-[#222] w-96 p-6 rounded mt-6'>
        <h2 className='text-center text-3xl'>{text}</h2>
      </div>
    </div>
  )
}
