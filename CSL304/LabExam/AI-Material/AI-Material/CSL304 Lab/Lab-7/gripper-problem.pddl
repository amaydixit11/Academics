;; Gripper Problem Instance
(define (problem gripper-problem)
  (:domain gripper-domain)
  
  ;; Objects
  (:objects
    rooma roomb - room
    ball1 ball2 ball3 ball4 - ball
    left right - gripper
  )
  
  ;; Initial State
  (:init
    ;; Robot starts in room A
    (at-robby rooma)
    
    ;; All balls start in room A
    (at-ball ball1 rooma)
    (at-ball ball2 rooma)
    (at-ball ball3 rooma)
    (at-ball ball4 rooma)
    
    ;; Both grippers are initially free
    (free left)
    (free right)
  )
  
  ;; Goal State
  (:goal (and
    (at-ball ball1 roomb)
    (at-ball ball2 roomb)
    (at-ball ball3 roomb)
    (at-ball ball4 roomb)
  ))
)