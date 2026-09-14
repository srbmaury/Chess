import { expect, test } from 'vitest'

import './styles.css'


test('base form controls keep readable dark-theme colors', () => {
  const select = document.createElement('select')
  select.innerHTML = '<option>Example</option>'
  const input = document.createElement('input')
  document.body.append(select, input)

  const selectStyle = getComputedStyle(select)
  const inputStyle = getComputedStyle(input)

  expect(selectStyle.backgroundColor).toBe('rgb(17, 19, 15)')
  expect(selectStyle.color).toBe('rgb(243, 244, 239)')
  expect(inputStyle.backgroundColor).toBe('rgb(17, 19, 15)')
  expect(inputStyle.color).toBe('rgb(243, 244, 239)')
})
