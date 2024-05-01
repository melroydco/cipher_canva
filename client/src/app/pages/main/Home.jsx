import { Link } from 'react-router-dom'

export const Home = () => {
  return (
    <div className='flex-1 flex items-center justify-center'>
      <div className='flex flex-col items-center max-w-6xl m-auto gap-2'>
        <h1 className='title-text-gradient text-6xl'>CipherCanva</h1>
        <p className='text-center text-xl opacity-80'>
          Embark on a journey into the fascinating realm of covert communication
          and digital concealment. Welcome to our Steganography Hub, where the
          art of hidden messages unfolds, and a vibrant community thrives!
        </p>
        <Link
          className='bg-[#555] p-5 rounded-lg border border-white text-lg mt-12'
          to='/encode'
        >
          Lets get started
        </Link>
      </div>
    </div>
  )
}
