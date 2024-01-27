# Human Abstraction Gymnasium Language (HAGL)
Goal of this library is to simplify work with environment. In some times when you create environment there are 
necessity to use Box space because of library support, simplicity and etc. For example
```
Box(-1, 1, (20,))
```
And then, when implementing step method work with observation as numerical array:
```
position = observation[0:2]
```
Such work could be very uncomfortable. One of the existed solution is to use Dict space:
```
Dict({
    "position": Box(-1, 1, (2,)),
    "velocity": Box(-1, 1, (2,))
})
```
Dict can be transformed in Box space with flatten, unflatten and flatten_space functions from Gymnasium.
With HAGL it can be written in such way:
```
class Observation:
    position = Position()
    velocity = Velocity()
```
This allows to use objects in step function:
```
position = observation.position
```
And then, using compile_type function or HAGLWrapper (for Gymnasium), HAGLParallelWrapper (for PettingZoo) this will be
transformed to Box space. Difference of HAGL and Dict is that:
* Can add method to class and use it in step function
* Can use inheritance
* When describe spaces with HAGL there are possibility to use usual python types for float, bool and arrays
* There are some built-in types, for example Position and Velocity. 

As a result: HAGL can be used to automate transformation of Gymnasium space representation to Python object