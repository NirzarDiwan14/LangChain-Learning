from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

react_texts = [
    "React is a JavaScript library used for building user interfaces. It was developed by Facebook and is widely used to create single-page applications (SPAs). React allows developers to build reusable UI components that can efficiently update in response to changes in data.",
    "In React, everything is a component. Components are the building blocks of a React application. They can be either: Class components (using ES6 classes), Functional components (using functions). Components allow developers to create reusable, self-contained pieces of the UI.",
    "JSX is a syntax extension to JavaScript that allows you to write HTML elements inside JavaScript. JSX makes the code more readable and is transformed into JavaScript functions using tools like Babel. Example: const element = <h1>Hello, world!</h1>.",
    "State is a built-in React object that is used to store data that affects the rendering of a component. State is mutable and can be updated via setState() (in class components) or useState() (in functional components).",
    "Props are short for 'properties' and are used to pass data between components. Props are immutable and are read-only within the component receiving them. Props are passed to components like function arguments. Example: function Welcome(props) { return <h1>Hello, {props.name}</h1>; }.",
    "In class components, React provides lifecycle methods to run code at specific points in a component’s life cycle. Some common lifecycle methods include: componentDidMount(), componentDidUpdate(), componentWillUnmount(). These methods help with tasks such as data fetching or cleanup.",
    "useEffect is a hook in functional components that allows side effects, such as data fetching, subscriptions, and manual DOM manipulation, to occur. It's similar to the lifecycle methods in class components. Example: useEffect(() => { console.log('Component mounted'); return () => { console.log('Component unmounted'); }; }, []);.",
    "useState is a hook that allows functional components to have local state. It returns an array where the first element is the current state and the second element is the function to update the state. Example: const [count, setCount] = useState(0); const increment = () => setCount(count + 1);.",
    "The Context API allows React components to share data without passing props down manually at every level. It’s useful for managing global state like authentication, themes, etc.",
    "Redux is a state management library that helps manage the state of your application in a predictable way. React Redux is a binding for using Redux with React.",
    "React encourages building complex UIs by composing simple components. This approach allows you to break down large applications into smaller, manageable pieces, making the code more maintainable.",
    "React hooks are functions that allow you to use state and lifecycle features in functional components. Some common hooks include: useState(), useEffect(), useContext(), useReducer().",
    "React Fragments allow you to group a list of children without adding extra nodes to the DOM. Example: return (<><h1>Hello</h1><h2>World</h2></>);.",
    "Code splitting is the process of splitting your application into smaller bundles, so only the necessary code is loaded when a user visits a page. This improves the performance of your application. React provides built-in support for code splitting using React.lazy() and Suspense.",
    "Error Boundaries are components that catch JavaScript errors anywhere in their child component tree, log those errors, and display a fallback UI. Example: class ErrorBoundary extends React.Component { constructor(props) { super(props); this.state = { hasError: false }; } static getDerivedStateFromError(error) { return { hasError: true }; } componentDidCatch(error, info) { console.log(error, info); } render() { if (this.state.hasError) { return <h1>Something went wrong.</h1>; } return this.props.children; } }.",
    "React provides tools to optimize performance, such as: React.memo() for memoizing components, useMemo() for memoizing values, useCallback() for memoizing functions.",
    "Concurrent Mode is a set of new features in React that help apps stay responsive and gracefully adjust to the user’s device capabilities and network speed.",
    "React DevTools is a browser extension that allows you to inspect the React component tree, state, props, and performance of your React app. It’s an essential tool for debugging and optimizing React applications.",
    "React simplifies the development of intelligent applications by abstracting complex workflows into reusable components. Its modular design allows developers to focus on building logic rather than handling low-level LLM details."
]


embedding_model= HuggingFaceEndpointEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_texts(
    texts=react_texts,
    embedding=embedding_model,
    collection_name="Chroma-demo-lecture",
    persist_directory="./chroma_db"
)
query = "Tell me about React hooks"
results = vector_store.similarity_search(query,k=3)

# Print the results
for result in results:
    print(f"Result: {result.page_content}")
