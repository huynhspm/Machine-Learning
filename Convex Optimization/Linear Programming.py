# Một anh nông dân có tổng cộng 10ha (10 hecta) đất canh tác. Anh dự tính trồng cà phê 
# và hồ tiêu trên số đất này với tổng chi phí cho việc trồng này là không quá 16T (triệu đồng). 
# Chi phí để trồng cà phê là 2T cho 1ha, để trồng hồ tiêu là 1T/ha/. Thời gian trồng 
# cà phê là 1 ngày/ha và hồ tiêu là 4 ngày/ha; trong khi anh chỉ có thời gian tổng cộng là 32 ngày. 
# Sau khi trừ tất cả các chi phí (bao gồm chi phí trồng cây), mỗi ha cà phê mang lại lợi nhuận 5T, 
# mỗi ha hồ tiêu mang lại lợi nhuận 3T. Hỏi anh phải trồng như thế nào để tối đa lợi nhuận?

# Gọi x và y lần lượt là số ha cà phê và hồ tiêu mà anh nông dân nên trồng. 
# --> profit: f(x ,y) = 5x + 3y (triệu đồng)
# Constrain: 
# Area <= 10: x + y <= 10
# Cost <= 16T: 2x + y <= 16
# Time <= 32 days: x + 4y <= 32
# Area >= 0: x, y >= 0

# (x, y) = argmax(5x + 3y) subject to: x + y <= 1; 2x + y <= 16; x + 4y <= 32; x, y >= 0 

from cvxopt import matrix, solvers
c = matrix([-5., -3.])
G = matrix([[1., 2., 1., -1., 0.], [1., 1., 4., 0., -1.]])
h = matrix([10., 16., 32., 0., 0.])

solvers.options['show_progress'] = False
sol = solvers.lp(c, G, h)

print('Solution"')
print(sol['x'])