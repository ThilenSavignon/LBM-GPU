# Group Participants
- Jules SEBAN (jules.seban@etu.univ-grenoble-alpes.fr)
- Thilen SAVIGNON (thilen.savignon@etu.univ-grenoble-alpes.fr)

# Exercises
## Exercise 1
We completed the exercise by specifying how the global domain should run 
in the init function and implementing the communication in the exchange function. 
The exchange function has a stack-like order with the send-receive components, 
we need this structure to avoid being in a deadlock situation.

There is no known bug to us for this exercise.

## Exercise 2
In this exercise, we improve the communication protocol with the Odd-Even implementation.
Every process with an even rank executes the following sequence of communications:
- Receive from the left
- Receive from the right
- Send to the right
- Send to the left
Every odd process will complete the exact inverse of this sequence (in both left/right and send/receive sense).
This allows for a better parallelisation of the tasks, with less waiting overall.

There is no known bug to us for this exercise.

## Exercise 3
We finalize the 1D-split implementation in this exercise by switching from blocking to non-blocking communications.
This further reduces wait times by buffering communication, allowing a process to communicate with multiple processes at once.

There is no known bug to us for this exercise.

## Exercise 4
We initialized all the values of `comm` and allocated the ressources needed (such as buffers), 
we also created the Cartesian space using the provided MPI function. 
When communicating information in the exchange function, 
we kept the odd-even code structure while using the buffers for exchanging rows in the correct order.
Columns are exchanged normally using a simple refactor of the previous code. Note that we do not
explicitely communicate the corners, simply because the size of our buffers are such that they include them.

There is no known bug to us for this exercise.

## Exercise 5
The initialization of this exercise is reusing the one of exercise 4, adding the definition of our datatype on top. 
The exchange function is also very similar, we simply modify the previous code to incorporate the datatypes and
remove our memory copies in and out of the buffers because MPI will take care of that by itself. Note that the buffers we
allocate in exercice 4 are also allocated here, but used by neither us nor MPI.

There is no known bug to us for this exercise.

## Exercise 6
Again a very similar initialization to exercise 4 and 5, except we commit more datatypes for edge cases within borders.
Indeed, now that our send and receive functions are asynchronous, they may interfere with each other and the corners of
some zones may be set to uninitialized values. For this reason, we seperate the communication of borders and corners,
and we dynamically change the size of our datatypes and buffers depending on the location of the zone
(whether it is on a side or not).

There is no known bug to us for this exercise.

