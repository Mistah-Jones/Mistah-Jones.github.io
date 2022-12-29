// This function takes the translation and two rotation angles (in radians) as input arguments.
// The two rotations are applied around x and y axes.
// It returns the combined 4x4 transformation matrix as an array in column-major order.
// You can use the MatrixMult function defined in project5.html to multiply two 4x4 matrices in the same format.
function GetModelViewMatrix( translationX, translationY, translationZ, rotationX, rotationY )
{
	var translation = [
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		translationX, translationY, translationZ, 1
	];

	var rollMatrix = [
		1, 0, 0, 0,
		0, Math.cos( rotationX ), Math.sin( rotationX ), 0,
		0, -Math.sin( rotationX ), Math.cos( rotationX ), 0,
		0, 0, 0, 1
	];

	var pitchMatrix = [
		Math.cos( rotationY ), 0, -Math.sin( rotationY ), 0,
		0, 1, 0, 0,
		Math.sin( rotationY ), 0, Math.cos( rotationY ), 0,
		0, 0, 0, 1
	];

	// var mvp = MatrixMult( projectionMatrix, translation );
	var mvp = MatrixMult( translation, pitchMatrix );
	mvp = MatrixMult( mvp, rollMatrix );
	return mvp;
}


// [TO-DO] Complete the implementation of the following class.

class MeshDrawer
{
	// The constructor is a good place for taking care of the necessary initializations.
	constructor()
	{
		// Compile the shader program
		this.prog = InitShaderProgram( meshVS, meshFS );

		this.showTexButton = true;

		// Get the ids of the uniform variables in the shaders
		this.mvp = gl.getUniformLocation( this.prog, 'mvp' );
		this.mv = gl.getUniformLocation( this.prog, 'mv' );
		this.mNorm = gl.getUniformLocation( this.prog, 'mNorm' ); 
		this.swapped = gl.getUniformLocation( this.prog, 'swapped' );
		this.lightDirection = gl.getUniformLocation( this.prog, 'lightDirection' );
		this.viewerDirection = gl.getUniformLocation( this.prog, 'viewerDirection');
		this.tex = gl.getUniformLocation( this.prog, 'tex' );
		this.showTex = gl.getUniformLocation( this.prog, 'showTex' );
		this.shininess = gl.getUniformLocation( this.prog, 'shininess' );

		// Get the id of the vertex attribute in the shaders
		this.pos = gl.getAttribLocation( this.prog, 'pos' );
		this.txc = gl.getAttribLocation( this.prog, 'txc' );
		this.normal = gl.getAttribLocation( this.prog, 'normal' );

		// Create the buffer objects
		this.vertexBuffer = gl.createBuffer();
		this.normalBuffer = gl.createBuffer();
		this.lineBuffer = gl.createBuffer();
		this.textureBuffer = gl.createBuffer();
	}
	
	// This method is called every time the user opens an OBJ file.
	// The arguments of this function is an array of 3D vertex positions,
	// an array of 2D texture coordinates, and an array of vertex normals.
	// Every item in these arrays is a floating point value, representing one
	// coordinate of the vertex position or texture coordinate.
	// Every three consecutive elements in the vertPos array forms one vertex
	// position and every three consecutive vertex positions form a triangle.
	// Similarly, every two consecutive elements in the texCoords array
	// form the texture coordinate of a vertex and every three consecutive 
	// elements in the normals array form a vertex normal.
	// Note that this method can be called multiple times.
	setMesh( vertPos, texCoords, normals )
	{
		// Update the contents of the vertex buffer objects.
		this.numTriangles = vertPos.length / 3;

		// Bind the program
		gl.useProgram( this.prog );

		// Bind the vertex buffer to send vertex information to GPU
		gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
		// Send the data
		gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( vertPos ), gl.STATIC_DRAW );

		gl.bindBuffer( gl.ARRAY_BUFFER, this.normalBuffer );
		gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( normals ), gl.STATIC_DRAW  )

		gl.bindBuffer( gl.ELEMENT_ARRAY_BUFFER, this.lineBuffer );
		gl.bufferData( gl.ELEMENT_ARRAY_BUFFER, new Float32Array( vertPos ), gl.STATIC_DRAW );

		gl.bindBuffer( gl.ARRAY_BUFFER, this.textureBuffer );
		gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( texCoords ), gl.STATIC_DRAW );
	}
	
	// This method is called when the user changes the state of the
	// "Swap Y-Z Axes" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	swapYZ( swap )
	{
		gl.useProgram( this.prog );
		gl.uniform1i( this.swapped, swap );
	}
	
	// This method is called to draw the triangular mesh.
	// The arguments are the model-view-projection transformation matrixMVP,
	// the model-view transformation matrixMV, the same matrix returned
	// by the GetModelViewProjection function above, and the normal
	// transformation matrix, which is the inverse-transpose of matrixMV.
	draw( matrixMVP, matrixMV, matrixNormal )
	{
		// Bind the program and set the vertex attribute
		gl.useProgram( this.prog );

		// Set Uniform variables
		gl.uniformMatrix4fv( this.mvp, false, matrixMVP );
		gl.uniformMatrix4fv( this.mv, false, matrixMV );
		gl.uniformMatrix3fv( this.mNorm, false, matrixNormal );

		// Set position coordinate attribute
		gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
		gl.vertexAttribPointer( this.pos, 3, gl.FLOAT, false, 0, 0 );
		gl.enableVertexAttribArray( this.pos );

		// Set normals attribute (matrixNormal * normal)
		gl.bindBuffer( gl.ARRAY_BUFFER, this.normalBuffer );
		gl.vertexAttribPointer( this.normal, 3, gl.FLOAT, false, 0, 0 );
		gl.enableVertexAttribArray( this.normal );

		// Set texture coordinate attribute
		gl.bindBuffer( gl.ARRAY_BUFFER, this.textureBuffer );
		gl.vertexAttribPointer( this.txc, 2, gl.FLOAT, false, 0, 0 );
		gl.enableVertexAttribArray( this.txc );

		gl.bindBuffer( gl.ELEMENT_ARRAY_BUFFER, this.lineBuffer );

		// Draw the mesh
		gl.drawArrays( gl.TRIANGLES, 0, this.numTriangles );
	}
	
	// This method is called to set the texture of the mesh.
	// The argument is an HTML IMG element containing the texture data.
	setTexture( img )
	{		
		// Bind the texture
		const mytex = gl.createTexture();
		gl.bindTexture( gl.TEXTURE_2D, mytex );

		// Set the texture image data
		gl.texImage2D( gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img );
		gl.generateMipmap( gl.TEXTURE_2D );

		// Bind to Texture Unit
		gl.activeTexture( gl.TEXTURE0 );
		gl.bindTexture( gl.TEXTURE_2D, mytex );

		// Set uniform parameter of the fragment shader
		gl.useProgram( this.prog );
		gl.uniform1i( this.tex, 0 );
		
		// By default, when a new texture is set, we update the showTex variable to
		// dispaly the newly set texture if the box is clicked
		gl.useProgram( this.prog );
		gl.uniform1i( this.showTex, this.showTexButton );
	}
	
	// This method is called when the user changes the state of the
	// "Show Texture" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	showTexture( show )
	{		
		this.showTexButton = show;

		gl.useProgram( this.prog );
		gl.uniform1i( this.showTex, show );
	}
	
	// This method is called to set the incoming light direction
	setLightDir( x, y, z )
	{
		// Set the uniform parameter(s) of the fragment shader to specify the light direction.
		gl.useProgram( this.prog );
		gl.uniform3fv( this.lightDirection, new Float32Array( [x, y, z] ) );
	}
	
	// This method is called to set the shininess of the material
	setShininess( shininess )
	{
		// Set the uniform parameter(s) of the fragment shader to specify the shininess.
		gl.useProgram( this.prog );
		gl.uniform1f( this.shininess, shininess );
	}
}


// This function is called for every step of the simulation.
// Its job is to advance the simulation for the given time step duration dt.
// It updates the given positions and velocities.
function SimTimeStep( dt, positions, velocities, springs, stiffness, damping, particleMass, gravity, restitution )
{
	var forces = Array( positions.length ); // The total for per particle
	forces.fill(new Vec3(0, 0, 0));

	// [TO-DO] Compute the total force of each particle 

	// Gravity Force
	// F_G = m * g
	var gravForce = gravity.mul(particleMass);
	for (i = 0; i < forces.length; i++)
	{
		forces[i] = gravForce;
	}

	// Spring Force
	// F_0^s = k * (l - l_rest) * d_hat
	// l = |x_1 - x_0|
	// d_hat = (x_1 - x_0) / l
	for (i = 0; i < springs.length; i++)
	{
		var p0 = springs[i].p0;
		var p1 = springs[i].p1;
		var l_rest = springs[i].rest;
		var x0 = positions[p0].copy();
		var x1 = positions[p1].copy();
		var v0 = velocities[p0].copy();
		var v1 = velocities[p1].copy();

		var l = Math.sqrt(Math.pow(x1.x - x0.x, 2) + Math.pow(x1.y - x0.y, 2) + Math.pow(x1.z - x0.z, 2));

		var d_hat = (x1.sub(x0)).div(l);

		var sprForce = d_hat.mul( stiffness * (l - l_rest) );
		forces[p0] = forces[p0].add(sprForce);
		forces[p1] = forces[p1].sub(sprForce);

		// Apply Damping force
		var l_dot = (v1.sub(v0)).dot(d_hat);
		var dampForce = d_hat.mul(damping * l_dot);
		forces[p0] = forces[p0].add(dampForce);
		forces[p1] = forces[p1].sub(dampForce);
	}
	
	// [TO-DO] Update positions and velocities
	// (Semi-Implicit Euler Integration)
	// a_i <- f_i / m_i
		// a^(t) = F^(t) / m
	var accelerations = new Array( velocities.length ); 
	for (i = 0; i < accelerations.length; i++)
	{
		accelerations[i] = (forces[i].div(particleMass));
	}

	// v_i <- v_i + delta_t * a_i
		// v^(t + delta_t) = v^(t) + delta_t * a^(t)
	for (i = 0; i < velocities.length; i++)
	{
		velocities[i] = velocities[i].add(accelerations[i].mul(dt));
	}

	// x_i <- x_i + delta_t * v_i
		// x^(t + delta_t) = x^(t) + delta_t * v^(t + delta_t)
	for (i = 0; i < positions.length; i++)
	{
		//var newPos = positions[i].add(velocities[i].mul(dt));
		positions[i].inc(velocities[i].mul(dt));

		// Handle collisions
		// Y-coordinate collisions
		if (positions[i].y < -1)
		{
			var h = positions[i].y - (-1);
			positions[i].inc( new Vec3(0, Math.abs(restitution * h), 0) );

			velocities[i].scale(-restitution); 
		}
		if (positions[i].y > 1)
		{
			var h = positions[i].y - (1);
			positions[i].dec( new Vec3(0, Math.abs(restitution * h), 0) );

			velocities[i].scale(-restitution); 
		}
		// X-coordinate collisions
		if (positions[i].x < -1)
		{
			var h = positions[i].x - (-1);
			positions[i].inc( new Vec3(Math.abs(restitution * h), 0, 0) );

			velocities[i].scale(-restitution); 
		}
		if (positions[i].x > 1)
		{
			var h = positions[i].x - (1);
			positions[i].inc( new Vec3(-(restitution * h), 0, 0) );

			velocities[i].scale(-restitution); 
		}
		// Z-coordinate collisions
		if (positions[i].z < -1)
		{
			var h = positions[i].z - (-1);
			positions[i].inc( new Vec3( 0, 0, Math.abs(restitution * h) ) );

			velocities[i].scale(-restitution); 
		}
		if (positions[i].z > 1)
		{
			var h = positions[i].z - (1);
			positions[i].inc( new Vec3( 0, 0, -(restitution * h) ) );

			velocities[i].scale(-restitution); 
		}
	}
}

// Vertex shader source code
var meshVS = `
	attribute vec3 pos;
	attribute vec3 normal;
	attribute vec2 txc;

	uniform mat4 mvp;
	uniform mat4 mv;
	uniform mat3 mNorm;
	uniform bool swapped;

	varying vec3 surfaceNormal;
	varying vec2 texCoord;

	void main()
	{
		vec3 newPos = pos;

		if ( swapped )
		{
			newPos = vec3( pos.x, pos.z, pos.y );
		}

		surfaceNormal = mNorm * normalize( normal );
		texCoord = txc;

		gl_Position = mvp * vec4( newPos, 1 );
	}
`;

// Fragment shader source code
var meshFS = `
	precision mediump float;

	uniform vec3 lightDirection;
	uniform vec3 viewerDirection; // Normalized inverse of position.
	uniform float shininess;
	uniform sampler2D tex;
	uniform bool showTex;

	varying vec2 texCoord;
	varying vec3 surfaceNormal;

	void main()
	{
		// C = I (cos(theta)K_d + K_s(cos(phi))^omega)

		vec4 white = vec4( 1, 1, 1, 1 );
		vec4 k_d = vec4( 1, 1, 1, 1 );
		if ( showTex )
		{
			k_d = texture2D( tex, texCoord );			
		}

		// theta = n <dot> omega / ||n|| * ||omega||
		//float theta = dot( surfaceNormal, lightDirection ) / ( length( surfaceNormal ) * length( lightDirection ) );
		float cosTheta = dot( surfaceNormal, lightDirection );

		// h = omega + viewingDirection / | omega + viewingDirection |
		vec3 h = lightDirection + viewerDirection;
		h = normalize( h );

		// cos(phi) = n <dot> h
		float cosPhi = dot( surfaceNormal, h );

		vec4 c = white * ( cosTheta * k_d + ( white * pow(cosPhi, shininess) ) );

		gl_FragColor = c; //vec4(1,gl_FragCoord.z*gl_FragCoord.z,0,1);
	}
`;