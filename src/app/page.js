import LikeButton from './like-button';
import '../index.css'
function Header({ title }) {
  return <h1>{title ? title : 'Default title'}</h1>;
}
// what i want:
// hand selector from a grid: style when selected, reset button, presets button, etc.
// ability to choose amount of villains, 1-9
// ability to select range of hands that a given villain is playing
// tell amount of times you hit certain kinds of hands, 
export default function HomePage() {
  const names = ['Ada Lovelace', 'Grace Hopper', 'Margaret Hamilton'];
 
  return (
    <div>
      <Header title="Develop. Preview. Ship." />
      <ul>
        {names.map((name) => (
          <li key={name}>{name}</li>
        ))}
      </ul>
      <LikeButton />
    </div>
  );
}