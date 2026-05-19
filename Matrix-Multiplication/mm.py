class Matrix():
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.columns = len(data[0])

    def dot(self, other):
        if self.columns != other.rows:
            print(self.data)
            print(other.data)
            raise ValueError("Shape Mismatch.")
        
        #     3x3     3x3
        #
        #   [1,2,3] [1,2,3]
        #   [4,5,6] [4,5,6]
        #   [7,8,9] [7,8,9]
        #
        #    self    other
        #

        result = []
        for i in range(self.rows):
            row = []
            for j in range(other.columns):
                product = 0
                for k in range(self.columns):
                    product += self.data[i][k] * other.data[k][j]
                row.append(product)
            result.append(row)
        return Matrix(result)
    

    def __add__(self, other):
        if self.rows != other.rows or self.columns != other.columns:
            raise ValueError("Shape Mismatch")
        
        #    2x3     2x3
        #
        #   [1,2,3], [1,2,3],
        #   [3,4,2]  [3,4,2]
        #
        #    self other
        #

        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.columns):
                add = self.data[i][j] + other.data[i][j]
                row.append(add)
            result.append(row)
        return Matrix(result)
    
    
    def transpose(self):
        
        #   [1,2,3]
        #   [3,4,2]

        # output:
        # [1,3]
        # [2,4]
        # [3,2]
        #

        result = []

        for i in range(self.columns):
            row = []
            for j in range(self.rows):
                row.append(self.data[j][i])
            result.append(row)
        return Matrix(result)

    def __repr__(self):
        rows = []
        for i in range(self.rows):
            rows.append(str(self.data[i]))

        return "\n".join(rows)





# m1 = Matrix([[1],[2],[3]])
m1 = Matrix([[1,2,3],[3,4,2]])
m2 = Matrix([[1,2],[3,4], [3,4]])

# result = m1.dot(m2)
# result = m1 + m2

result = m1.dot(m2).transpose()
print(result)

# print(m1)