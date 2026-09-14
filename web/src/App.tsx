import { useState } from 'react'

import CommunityControls from './CommunityControls'
import LegacyApp from './LegacyApp'

export default function App() {
  const [profileRevision, setProfileRevision] = useState(0)
  return <>
    <CommunityControls onProfileChanged={() => setProfileRevision((value) => value + 1)} />
    <LegacyApp key={profileRevision} />
  </>
}
