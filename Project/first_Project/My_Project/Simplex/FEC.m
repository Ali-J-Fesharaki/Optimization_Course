function [function_value]=FEC(f,x,x_s)
    global FE FE_list;
    FE=FE+1;
    function_value=vpa(subs(f,x,x_s));
end