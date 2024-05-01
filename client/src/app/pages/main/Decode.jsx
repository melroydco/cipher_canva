import { useMutation } from '@tanstack/react-query'
import clsx from 'clsx'
import React, { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

export const Decode = () => {
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const imageInputRef = useRef(null)
  const imageUrl = file ? URL.createObjectURL(file) : ''

  const mutation = useMutation({
    mutationFn: formData =>
      fetch('http://127.0.0.1:3030/image/decode', {
        method: 'POST',
        body: formData
      }).then(res => res.json()),
    onSuccess: data => {
      navigate(`/decode/success?text=${data.text}`)
    }
  })

  function handleSubmit (evt) {
    evt.preventDefault()
    mutation.mutate(new FormData(evt.target))
  }

  const handleFileInputChange = event => {
    const file = event.target.files[0]
    setFile(file)
  }

  function handleRemoveImage () {
    setFile(null)
    imageInputRef.current.value = null
  }
  return (
    <form
      onSubmit={handleSubmit}
      className='flex-1 flex flex-col items-center justify-center gap-12'
    >
      <div className='flex flex-col gap-4'>
        <h1 className='title-text-gradient text-center text-6xl'>Decoder</h1>
        <p className='text-center'>
          Decode the secrets within! Upload the encoded image and reveal the
          hidden message in our steganography page.
        </p>
      </div>
      <label
        className={clsx(!file ? 'flex' : 'hidden', 'flex-col cursor-pointer')}
      >
        <span>Upload image</span>
        <img src='/uploadIcon.png' alt='upload' width={100} />
        <input
          type='file'
          name='image'
          className='h-0 w-0'
          ref={imageInputRef}
          onChange={handleFileInputChange}
        />
      </label>
      <div className={clsx(file ? 'flex' : 'hidden', 'flex-col gap-6')}>
        <div className='flex flex-col gap-4'>
          <img src={imageUrl} alt={file?.name} width={300} />
          <button type='button' onClick={handleRemoveImage}>
            Remove
          </button>
        </div>
        <button
          type='submit'
          className='bg-[#222] py-4 px-12 rounded'
          disabled={mutation.isPending}
        >
          {mutation.isPending ? 'Decoding...' : 'Decode'}
        </button>
      </div>
    </form>
  )
}
