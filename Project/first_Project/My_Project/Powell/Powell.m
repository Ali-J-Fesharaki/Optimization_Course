   f   u   n   c   t   i   o   n       [   o   p   t   i   m   u   m   _   p   o   i   n   t   ,   o   p   t   i   m   u   m   _   v   a   l   u   e   ,   p   o   w   e   l   l   _   i   t   e   r   a   t   i   o   n   ]       =       P   o   w   e   l   l   (   f   ,   x   ,   x   0   ,   o   r   d   e   r   ,   t   o   l   _   p   o   w   e   l   l   ,   t   o   l   _   G   S   ,   N   )      
   f   o   r   m   a   t       l   o   n   g      
   s   y   m   s       a   l   f      
   v   (   :   ,   1   )       =       x   0   ;      
   n   =   s   i   z   e   (   x   0   ,   1   )   ;      
   F   u   n   c   _   e   v   a   l   (   1   ,   :   )   =   0   ;      
   i   t   e   r   (   1   ,   :   )   =   0   ;      
   f   u   n   c   _   e   v   a   l   _   G   S   =   0   ;      
   j   =   0   ;      
   e   p       =       t   o   l   _   p   o   w   e   l   l   ;      
   F   E       =       0   ;      
   m   a   x   _   c   y   c   l   e   =   1   5   0   ;      
      
   s       =       e   y   e   (   n   ,   n   )   ;       %   t   h   i   s       c   a   n       g   e   n   e   r   a   t   e       d   i   f   f   e   r   e   n   t       d   i   r   e   c   t   i   o   n   s       a   l   o   n   g       c   o   o   r   d   i   n   a   t   e       a   x   e   s          
   d   (   :   ,   1   ,   1   )       =       s   (   :   ,   n   )   ;      
   f   o   r       t   =   2   :   n   +   1      
                   d   (   :   ,   t   ,   1   )       =       s   (   :   ,   t   -   1   )   ;      
   e   n   d      
      
   f   o   r       c   y   c   l   e   =   1   :   m   a   x   _   c   y   c   l   e      
           f   p   r   i   n   t   f   (   '       *   *   *   *   *           c   y   c   l   e   :       %   i       *   *   *   *   *   \   n   '   ,   c   y   c   l   e   )      
           i   f       c   y   c   l   e   >   1      
                   k   =   j   -   n   +   1   ;      
                   s   (   :   ,   1   )       =       v   (   :   ,   j   +   1   )       -       v   (   :   ,   k   )   ;      
                   d   (   :   ,   n   +   1   ,   c   y   c   l   e   )       =       s   (   :   ,   1   )   ;      
                   f   o   r       t   =   n   -   1   :   n      
                                   d   (   :   ,   t   ,   c   y   c   l   e   )       =       d   (   :   ,   t   +   1   ,   c   y   c   l   e   -   1   )   ;      
      
                                   i   f       m   o   d   (   c   y   c   l   e   ,   n   )   =   =   1      
                                                   s       =       e   y   e   (   n   ,   n   )   ;      
                                                   d   (   :   ,   1   ,   c   y   c   l   e   )       =       s   (   :   ,   n   )   ;      
                                                   f   o   r       t   =   2   :   n   +   1      
                                                               d   (   :   ,   t   ,   c   y   c   l   e   )       =       s   (   :   ,   t   -   1   )   ;      
                                                   e   n   d      
                                   e   n   d          
      
                                   i   f       n   o   r   m   (   d   (   :   ,   t   ,   c   y   c   l   e   )   )   >   1              
                                                   d   (   :   ,   t   ,   c   y   c   l   e   )       =       d   (   :   ,   t   ,   c   y   c   l   e   )   /   n   o   r   m   (   d   (   :   ,   t   ,   c   y   c   l   e   )   )   ;      
                                   e   n   d              
                  
                   e   n   d      
                   d   (   :   ,   1   ,   c   y   c   l   e   )   =   d   (   :   ,   n   +   1   ,   c   y   c   l   e   )   ;      
                   %   S   p   =   d   (   :   ,   :   ,   c   y   c   l   e   )      
           e   n   d      
      
           f   o   r       i   =   1   :   n   +   1          
                           i   =   i   +   j   ;      
                           i   t   e   r   (   i   ,   :   )   =   i   -   1   ;      
                           F   (   i   -   j   ,   c   y   c   l   e   )       =       v   p   a   (   s   u   b   s   (   f   ,   x   ,   v   (   :   ,   i   )   )   ,   5   )   ;       %   f   u   c   t   i   o   n       e   v   a   l   u   a   t   i   o   n      
                           x   _   m   i   n   u   s   _   e   p       =       v   (   :   ,   i   )       -       e   p   *   d   (   :   ,   i   -   j   ,   c   y   c   l   e   )   ;       %   x   ^   +      
                           x   _   p   l   u   s   _   e   p           =       v   (   :   ,   i   )       +       e   p   *   d   (   :   ,   i   -   j   ,   c   y   c   l   e   )   ;       %   x   ^   -      
                           F   _   m   i   n   u   s   _   e   p       =       (   v   p   a   (   s   u   b   s   (   f   ,   x   ,   x   _   m   i   n   u   s   _   e   p   )   )   )   ;       %   f   ^   +      
                           F   _   p   l   u   s   _   e   p           =       (   v   p   a   (   s   u   b   s   (   f   ,   x   ,   x   _   p   l   u   s   _   e   p   )   )   )   ;       %   f   ^   +      
                           F   E       =       F   E       +       n   ;       %   c   o   u   n   t   e   r       o   f       f   u   n   c   t   i   o   n       e   v   a   l   u   a   t   i   o   n       f   (   x   )       &       f   ^   -       &       f   ^   +      
                           F   u   n   c   _   e   v   a   l   (   i   ,   :   )   =   F   E   ;      
                           i   f       F   _   p   l   u   s   _   e   p       <       F   (   i   -   j   ,   c   y   c   l   e   )      
                                           x   _   a   l       =       v   (   :   ,   i   )       +       a   l   f   *   d   (   :   ,   i   -   j   ,   c   y   c   l   e   )   ;      
                           e   l   s   e      
                                           x   _   a   l       =       v   (   :   ,   i   )       -       a   l   f   *   d   (   :   ,   i   -   j   ,   c   y   c   l   e   )   ;      
                           e   n   d      
                                   F   _   a   l       =       v   p   a   (   s   u   b   s   (   f   ,   x   ,   x   _   a   l   )   )   ;      
                                   i   f       o   r   d   e   r       >       2      
                                                   [   a   l   f   a   _   f   i   n   a   l   ,   F   E   _   G   S   ]       =       g   o   l   d   e   n   _   s   e   a   r   c   h   (   F   _   a   l   ,   t   o   l   _   G   S   ,   N   )   ;      
                                                   a   l   f   a   =   v   p   a   (   a   l   f   a   _   f   i   n   a   l   ,   1   2   )   ;      
                                                   f   u   n   c   _   e   v   a   l   _   G   S   =   f   u   n   c   _   e   v   a   l   _   G   S   +   F   E   _   G   S   ;      
                                                   f   p   r   i   n   t   f   (   '   \   n   s   u   m       o   f       t   h   e       g   o   l   d   e   n       s   e   a   r   c   h       e   v   a   l   u   a   t   i   o   n       i   s   :       %   i       \   n   '   ,   f   u   n   c   _   e   v   a   l   _   G   S   )      
                                                   G   S   _   i   t   e   r   a   t   i   o   n   (   i   ,   :   )   =   F   E   _   G   S   ;      
                                   e   l   s   e      
                                                   a   l   f   a   =   v   p   a   (   s   o   l   v   e   (   d   i   f   f   (   s   u   b   s   (   f   ,   x   ,   x   _   a   l   )   ,   a   l   f   )   =   =   0   ,   a   l   f   )   ,   1   2   )   ;      
                                   e   n   d      
                                   i   f       F   _   p   l   u   s   _   e   p       <       F   (   i   -   j   ,   c   y   c   l   e   )      
                                                   v   (   :   ,   i   +   1   )       =       v   (   :   ,   i   )       +       a   l   f   a   *   d   (   :   ,   i   -   j   ,   c   y   c   l   e   )   ;      
                                   e   l   s   e      
                                                   v   (   :   ,   i   +   1   )       =       v   (   :   ,   i   )       -       a   l   f   a   *   d   (   :   ,   i   -   j   ,   c   y   c   l   e   )   ;      
                                   e   n   d      
           e   n   d      
                   i   f       F   _   p   l   u   s   _   e   p       >       F   (   i   -   j   ,   c   y   c   l   e   )       &   &       F   _   m   i   n   u   s   _   e   p       >       F   (   i   -   j   ,   c   y   c   l   e   )       &   &       (   n   o   r   m   (   v   (   :   ,   i   +   1   )   -   v   (   :   ,   i   )   )   <   t   o   l   _   p   o   w   e   l   l   )      
                           d   i   s   p   l   a   y   (   '   P   r   o   b   l   e   m       s   o   l   v   e   d   '   )      
                           b   r   e   a   k      
                   e   n   d      
           j   =   i   ;      
              
   e   n   d      
   f   u   n   c   t   i   o   n   _   e   v   a   l   u   a   t   i   o   n   s   =   F   u   n   c   _   e   v   a   l   (   i   ,   :   )      
   o   p   t   i   m   u   m   _   p   o   i   n   t       =   v   (   :   ,   i   )   ;      
   o   p   t   i   m   u   m   _   v   a   l   u   e       =       v   p   a   (   s   u   b   s   (   f   ,   x   ,   o   p   t   i   m   u   m   _   p   o   i   n   t   )   ,   5   )   ;      
      
   %   %       s   a   v   i   n   g       R   e   s   u   l   t   s      
   G   S   _   i   t   e   r   a   t   i   o   n   (   i   ,   :   )   =   0   ;      
   x       =       v   p   a   (   v   ,   5   )   '   ;      
   p   o   w   e   l   l   _   i   t   e   r   a   t   i   o   n   =   (   0   :   i   -   1   )   '   ;      
      
                   R   e   s   u   l   t   s   T   a   b   l   e       =       t   a   b   l   e   (   p   o   w   e   l   l   _   i   t   e   r   a   t   i   o   n   ,       G   S   _   i   t   e   r   a   t   i   o   n   )   ;      
                   f   o   r       i       =       1   :   n      
                                   R   e   s   u   l   t   s   T   a   b   l   e   .   (   s   p   r   i   n   t   f   (   '   x   %   d   '   ,       i   )   )       =       d   o   u   b   l   e   (   x   (   :   ,       i   )   )   ;      
                   e   n   d      
      
                   %       S   a   v   e       t   h   e       t   a   b   l   e       t   o       a   n       E   x   c   e   l       f   i   l   e       w   i   t   h   o   u   t       h   e   a   d   e   r   s      
                   w   r   i   t   e   t   a   b   l   e   (   R   e   s   u   l   t   s   T   a   b   l   e   ,       '   p   o   w   e   l   l   _   i   t   e   r   a   t   i   o   n   .   x   l   s   x   '   )      
   e   n   d      
      
      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ��