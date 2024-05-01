export function About () {
  return (
    <div className='flex flex-col gap-12 max-w-6xl mx-auto'>
      <div className='flex flex-col gap-6'>
        <h1 className='title-text-gradient text-6xl text-center'>About Us</h1>
        <p className='text-center'>
          Embark on a journey into the fascinating realm of covert communication
          and digital concealment. Welcome to our Steganography Hub, where the
          art of hidden messages unfolds, and a vibrant community thrives!
        </p>
      </div>
      <div className='flex flex-col gap-12'>
        <h2 className='text-center title-text-gradient text-5xl'>
          Team Members
        </h2>
        <div className='flex gap-3 justify-between flex-wrap items-center'>
          <p>Anisha Sharal Dsouza</p>
          <p>Godson Jeevan Dsouza</p>
          <p>Lisha Princita Rodrigues Dsouza</p>
          <p>Melroy Dcosta</p>
        </div>
      </div>
    </div>
  )
}
