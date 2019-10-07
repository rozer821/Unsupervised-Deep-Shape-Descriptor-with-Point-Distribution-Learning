import numpy 
import torch
class PointVec():
    def init__(self,x,y,z,limit_scale):
        
        self.x = x
        self.y = y
        self.z = z
    def get_point_coord():
        return [self.x,self.y,self.z]
class PoissonDisk3D():
    def __init__(self,num_pc,limit_dis,limit_scale=[-1,1]):
        self.num_pc = num_pc
        self.limit_size = limit_size
        self.limit_dis = 22
        tries = 30;
        size = abs(limit_scale[1]-limit_scale[0])
        w = r/sqrt(3);
        x_grid_rows = floor(size/w);
        y_grid_cols = floor(size/w);
        z_grid_layers = floor(size/w);
        
        grid = torch.zeros(x_grid_rows,y_grid_cols,z_grid_layers)
        
        pos_ini = [
    // Add first point
    float x = size/2, y = size/2, z = size/2;
    PVector pos = new PVector(x, y, z);
    activeList.add(pos);
    grid[floor(x / w)][floor(y / w)][floor(z / w)] = pos;
    
    R = size/2;
  }
  
  
  int update(){
    for(int total=0; total<25; total++){
      if(activeList.size() > 0){
        int randIdx = floor(random(activeList.size()));
        PVector pos = activeList.get(randIdx); boolean found = false;
        for(int n=0; n<tries; n++){
          PVector sample = PVector.random3D();
          sample.setMag(random(r, 2*r));
          sample.add(pos);
          int col   = floor(sample.x / w);
          int row   = floor(sample.y / w);
          int layer = floor(sample.z / w);
          
          boolean insideSphere = (size/2 - sample.x)*(size/2 - sample.x) + 
                                 (size/2 - sample.y)*(size/2 - sample.y) + 
                                 (size/2 - sample.z)*(size/2 - sample.z) < R*R;
          
          // Check all neighboring cells
          boolean insideGrid = 0 <= row && row < rows && 
                               0 <= col && col < cols && 
                               0 <= layer && layer < layers;
          if(insideSphere && insideGrid && grid[row][col][layer] == null){
            boolean ok = true;
            for(int i=-1; i<=1; i++){ for(int j=-1; j<=1; j++){ for(int k=-1; k<=1; k++){
              if(0 <= row + i && row + i < rows && 
                 0 <= col + j && col + j < cols &&
                 0 <= layer + k && layer + k < layers){
                PVector neighbor = grid[row + i][col + j][layer + k];
                if(neighbor != null){
                  float d = PVector.dist(sample, neighbor);
                  if(d < r){ ok = false; }
                }
              }
            }}}
            if(ok){
              found = true;
              grid[row][col][layer] = sample;
              activeList.add(sample);
              break;
            }
          }
        }
        if(!found){ activeList.remove(randIdx); }
      }
    }
    return activeList.size();
  }