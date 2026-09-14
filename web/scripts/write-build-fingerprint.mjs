import { createHash } from 'node:crypto'
import { readdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true })
  const files = []
  for (const entry of entries) {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) files.push(...await walk(path))
    else if (entry.isFile()) files.push(path)
  }
  return files
}

const files = [
  ...await walk(join(root, 'src')),
  join(root, 'index.html'),
  join(root, 'package.json'),
].sort()

const hash = createHash('sha256')
for (const path of files) {
  hash.update(relative(root, path).replaceAll('\\', '/'))
  hash.update('\0')
  hash.update(await readFile(path))
  hash.update('\0')
}

await writeFile(join(root, 'dist', '.source-fingerprint'), `${hash.digest('hex')}\n`)
