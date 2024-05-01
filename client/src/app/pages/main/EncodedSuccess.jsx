import React, { useState, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import axios from 'axios';

export const EncodedSuccess = () => {
  const [searchParams] = useSearchParams()
  const encodedImageUrl = `http://localhost:3030${searchParams.get(
    'downloadUrl'
  )}`
  const fileInputRef = useRef(null)

  const handleImageUpload = () => {
    fileInputRef.current.click()
  }

  const handleFileInputChange = event => {
    const file = event.target.files[0]
    console.log('Selected file:', file)
    const imageUrl = URL.createObjectURL(file)
    setSelectedImage(imageUrl)
  }



  const EmailVerificationForm = () => {
    const [email, setEmail] = useState('');
    const [otpSent, setOtpSent] = useState(false);

    const handleSubmit = async (e) => {
      e.preventDefault();
      try {
        await axios.post('/sendOTP', { email });
        setOtpSent(true);
      } catch (error) {
        console.error('Error sending OTP:', error);
      }
    };
  }

  return (
    <><div className='flex-1 flex flex-col items-center justify-center gap-12'>
      <div className='flex flex-col gap-4'>
        <h1 className='text-center text-red-600 text-6xl title-text-gradient'>
          Encoder
        </h1>
        <p className='text-center'>
          Successfully encoded, your text is now seamlessly embedded into the
          image
        </p>
      </div>
      <img src={encodedImageUrl} width={360} />
      <a
        href={encodedImageUrl}
        target='_blank'
        className='py-4 px-12 bg-[#222] rounded'
      >
        Download
      </a>
      <div>
        <h1>Enter your email address</h1>
        {otpSent ? (
          <p>OTP sent to {email}. Please check your email.</p>
        ) : (
          <form onSubmit={handleSubmit}>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email"
              required />
            <button type="submit">Send OTP</button>
          </form>
        )}
      </div>
      </div></>
  )
};

