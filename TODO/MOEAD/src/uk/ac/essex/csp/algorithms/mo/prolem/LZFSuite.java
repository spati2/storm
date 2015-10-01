package uk.ac.essex.csp.algorithms.mo.prolem;

import static java.lang.Math.PI;
import static java.lang.Math.cos;
import static java.lang.Math.pow;
import static java.lang.Math.sin;
import static java.lang.Math.sqrt;

import java.util.ArrayList;
import java.util.List;

import uk.ac.essex.csp.algorithms.mo.ea.AbstractCMOProblem;

public abstract class LZFSuite extends AbstractCMOProblem {

	// control the shape
	void alphafunction(double alpha[], double[] x, int dim, int type){
		if(dim==2){
			if(type==21){
				alpha[0] = x[0];
				alpha[1] = 1 - sqrt(x[0]);
			}

			if(type==22){
				alpha[0] = x[0];
				alpha[1] = 1 - x[0]*x[0];
			}

			if(type==23){
				alpha[0] = x[0];
				alpha[1] = 1 - sqrt(alpha[0]) -
					alpha[0]*sin(10*alpha[0]*alpha[0]*PI);
			}
		}
		else
		{

			if(type==31){
				alpha[0] = cos(x[0]*PI/2)*cos(x[1]*PI/2);
				alpha[1] = cos(x[0]*PI/2)*sin(x[1]*PI/2);
				alpha[2] = sin(x[0]*PI/2);
			}

			if(type==32){
				alpha[0] = 1 - cos(x[0]*PI/2)*cos(x[1]*PI/2);
				alpha[1] = 1 - cos(x[0]*PI/2)*sin(x[1]*PI/2);
				alpha[2] = 1 - sin(x[0]*PI/2);
			}

			/*
			if(type==33){
			alpha[0] = x[0]*x[1];
			alpha[1] = x[0]*(1 - x[1]);
			alpha[2] = (1 - x[0]);
			}*/

			if(type==33){
				alpha[0] = x[0];
				alpha[1] = x[1];
				//alpha[2] = 3 - (x[0]*(1 + sin(3*pi*x[0])) + x[1]*(1 +	sin(3*pi*x[1])))/2;
				alpha[2] = 3 - (sin(3*PI*x[0]) + sin(3*PI*x[1])) - 2*(x[0] +
					x[1]);
			}
		}
	}
	
	double betafunction(double[] x, int type)
	{
		double beta = 0;
		int dim = x.length;
		if(dim==0) beta = 0;

		if(type==1){
			beta = 0;
			for(int i=0; i<dim; i++){
				beta+= x[i]*x[i];
			}
			beta = 2.0*beta/dim;
		}


		/*
		if(type==2){
		double sum = 0, prod = 1, xx;
		for(int i=0; i<dim; i++){
		xx  = 2*x[i];
		sum+= xx*xx;
		prod*=cos(2*pi*xx/sqrt(i+1));
		}
		beta = 2.0*(sum - 2*prod + 2)/dim;
		}*/

		if(type==2){
			beta = 0;
			for(int i=0; i<dim; i++){
				beta+= sqrt(i+1.0)*x[i]*x[i];
			}
			beta = 2.0*beta/dim;
		}

		if(type==3){
			double sum = 0, xx;
			for(int i=0; i<dim; i++){
				xx = 2*x[i];
				sum+= (xx*xx - cos(4*PI*xx) + 1);
			}
			beta = 2.0*sum/dim;
		}

		if(type==4){
			double sum = 0, prod = 1, xx;
			for(int i=0; i<dim; i++){
				xx  = 2*x[i];
				sum+= xx*xx;
				prod*=cos(10*PI*xx/sqrt(i+1.0));
			}
			beta = 2.0*(sum - 2*prod + 2)/dim;
		}
		return beta;
	}
	

	double linkfunc2(double x, double t1, int dim, int type, int css, int nvar){
		// type:  the type of curve
		// css:   the class of index
		double beta = 0;
		dim++;
		if(type==21){
			double xy   = 2*(x - 0.5);
			beta = xy - pow(t1, 0.5*(nvar + 3*dim - 8)/(nvar - 2));
		}

		if(type==22){
			double theta = 6*PI*t1 + dim*PI/nvar;
			double xy    = 2*(x - 0.5);
			beta = xy - sin(theta);
		}

		if(type==23){
			double theta = 6*PI*t1 + dim*PI/nvar;
			double ra    = 0.8*t1;
			double xy    = 2*(x - 0.5);
			if(css==1)
				beta = xy - ra*cos(theta);
			else{
				beta = xy - ra*sin(theta);
			}
		}

		if(type==24){
			double theta = 6*PI*t1 + dim*PI/nvar;
			double xy    = 2*(x - 0.5);
			double ra    = 0.8*t1;
			if(css==1)
				beta = xy - ra*cos(theta/3);
			else{
				beta = xy - ra*sin(theta);
			}
		}

		if(type==25){
			double rho   = 0.8;
			double phi   = PI*t1;
			double theta = 6*PI*t1 + dim*PI/nvar;
			double xy    = 2*(x - 0.5);
			if(css==1)
				beta = xy - rho*sin(phi)*sin(theta);
			else if(css==2)
				beta = xy - rho*sin(phi)*cos(theta);
			else
				beta = xy - rho*cos(phi);
		}

		if(type==26){
			double theta = 6*PI*t1 + dim*PI/nvar;
			double ra    = 0.3*t1*(t1*cos(4*theta) + 2);
			double xy    = 2*(x - 0.5);
			if(css==1)
				beta = xy - ra*cos(theta);
			else{
				beta = xy - ra*sin(theta);
			}
		}

		return beta;
	}
	
	double linkfunc3(double x, double t1, double t2, int dim, int type, int nvar){
		// type:  the type of curve
		// css:   the class of index
		double beta=0;
		dim++;

		if(type==31){
			double xy  = 4*(x - 0.5);
			double rate = 1.0*dim/nvar;
			beta = xy - 4*(t1*t1*rate + t2*(1.0-rate)) + 2;
		}

		//*
		if(type==32){
			double theta = 2*PI*t1 + dim*PI/nvar;
			double xy    = 4*(x - 0.5);
			//double zz    = sin(2*t2*pi);
			//double zz    = t2-1;
			beta = xy - 2*t2*sin(theta);
		}//*/

		return beta;
	}
	
	private List<Double> aa = new ArrayList<Double>();
	private List <Double> bb = new ArrayList<Double>();
	private List <Double> cc = new ArrayList<Double>();
	
	void objective(double[] x_var, double[] y_obj, int ltype, int dtype, int ptype)
	{
		int nobj = y_obj.length;
		int nvar = x_var.length;
		
		if(nobj==2)
		{
			if(ltype==21||ltype==22||ltype==23||ltype==24||ltype==26)
			{
				double g = 0, h = 0, a, b;
				aa.clear();
				bb.clear();
				
				for(int n=1;n<nvar;n++)
				{

					if(n%2==0){
						a = linkfunc2(x_var[n],x_var[0],n,ltype,1,nvar);  // linkage
						aa.add(a);
					}
					else
					{
						b = linkfunc2(x_var[n],x_var[0],n,ltype,2,nvar);
						bb.add(b);
					}
				}
				
				double[] aaarray = new double[aa.size()];
				for (int i=0;i<aaarray.length;i++) 
					aaarray[i]=aa.get(i);
				
				double[] bbarray = new double[bb.size()];
				for (int i=0;i<bbarray.length;i++) 
					bbarray[i]=bb.get(i);

				g = betafunction(aaarray, dtype);
				h = betafunction(bbarray, dtype);

				double[] alpha = new double[2];
				alphafunction(alpha,x_var,2,ptype);  // shape function
				y_obj[0] = alpha[0] + h;
				y_obj[1] = alpha[1] + g;
				aa.clear();
				bb.clear();
			}

			if(ltype==25)
			{
				double g = 0, h = 0, a, b;
				double e = 0, c;
				aa.clear();
				bb.clear();
				
				for(int n=1;n<nvar;n++){
					if(n%3==0){
						a = linkfunc2(x_var[n],x_var[0],n,ltype,1,nvar);  // linkage
						aa.add(a);
					}
					else if(n%3==1)
					{
						b = linkfunc2(x_var[n],x_var[0],n,ltype,2,nvar);
						bb.add(b);
					}
					else{
						c = linkfunc2(x_var[n],x_var[0],n,ltype,3,nvar);
						if(n%2==0)    aa.add(c);
						else          bb.add(c);
					}
				}
				
				double[] aaarray = new double[aa.size()];
				for (int i=0;i<aaarray.length;i++) 
					aaarray[i]=aa.get(i);
				
				double[] bbarray = new double[bb.size()];
				for (int i=0;i<bbarray.length;i++) 
					bbarray[i]=bb.get(i);
				
				g = betafunction(aaarray, dtype);          // distance function
				h = betafunction(bbarray, dtype);
				
				double[] alpha = new double[2];
				alphafunction(alpha,x_var,2,ptype);  // shape function
				
				y_obj[0] = alpha[0] + h;
				y_obj[1] = alpha[1] + g;
				aa.clear();
				bb.clear();
			}
		}


		if(nobj==3)
		{
			if(ltype==31||ltype==32)
			{
				double g = 0, h = 0, e = 0, a;
				aa.clear();
				bb.clear();
				cc.clear();
				
				for(int n=2;n<nvar;n++)
				{
					a = linkfunc3(x_var[n],x_var[0],x_var[1],n,ltype,nvar);
					if(n%3==0)			aa.add(a);
					else if(n%3==1)		bb.add(a);
					else				cc.add(a);
				}

				double[] aaarray = new double[aa.size()];
				for (int i=0;i<aaarray.length;i++) 
					aaarray[i]=aa.get(i);
				
				double[] bbarray = new double[bb.size()];
				for (int i=0;i<bbarray.length;i++) 
					bbarray[i]=bb.get(i);
				
				double[] ccarray = new double[cc.size()];
				for (int i=0;i<ccarray.length;i++) 
					ccarray[i]=cc.get(i);
				
				g = betafunction(aaarray, dtype);
				h = betafunction(bbarray, dtype);
				e = betafunction(ccarray, dtype);

				double alpha[] = new double[3];
				alphafunction(alpha,x_var,3,ptype);  // shape function
				y_obj[0] = alpha[0] + h;
				y_obj[1] = alpha[1] + g;
				y_obj[2] = alpha[2] + e;
				aa.clear();
				bb.clear();
				cc.clear();
			}
		}
	}
//	
//	private double[] idealpoint;
//	
//	@Override
//	public double[] getIdealPoint() {
//		if (idealpoint == null)
//			idealpoint = new double[] { 0, -1 };
//		return idealpoint;
//	}
	
	@Override
	protected void init() {
		this.domain = new double[this.parDimension][2];
		for (int i = 0; i < parDimension; i++) {
			domain[i][0] = 0;
			domain[i][1] = 1;
		}
		this.objDimension = 2;
		this.range = new double[objDimension][2];
		this.idealpoint = new double[] { 0, 0 };
	}
}
