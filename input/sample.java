package coms295;

public class Week13 {
	
	 class TreeNode {
	     int val;
		 TreeNode left;
		 TreeNode right;
		      TreeNode() {}
		      TreeNode(int val) { this.val = val; }
		      TreeNode(int val, TreeNode left, TreeNode right) {
		          this.val = val;
		          this.left = left;
		          this.right = right;
		      }
		  }
	
	/*
	 * Problem 100, Binary Trees
	 */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        
        if(p==null && q==null){
            return true;
        }
        if(p==null||q==null){
            return false;
        }
        if(p.val!=q.val){
            return false;
        }
        
        return isSameTree(p.left,q.left) && isSameTree(p.right,q.right);
        
    }
}
